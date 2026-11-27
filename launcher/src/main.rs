#![windows_subsystem = "windows"]

use std::env;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

#[cfg(windows)]
use std::os::windows::process::CommandExt;

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x08000000;

#[cfg(windows)]
#[link(name = "user32")]
extern "system" {
    fn MessageBoxW(hwnd: *mut core::ffi::c_void, text: *const u16, caption: *const u16, utype: u32) -> i32;
}

#[cfg(windows)]
fn show_error_dialog(title: &str, msg: &str) {
    let mut t_utf16: Vec<u16> = title.encode_utf16().collect();
    t_utf16.push(0);
    let mut m_utf16: Vec<u16> = msg.encode_utf16().collect();
    m_utf16.push(0);
    unsafe {
        MessageBoxW(core::ptr::null_mut(), m_utf16.as_ptr(), t_utf16.as_ptr(), 0x10);
    }
}

#[cfg(not(windows))]
fn show_error_dialog(title: &str, msg: &str) {
    eprintln!("{}: {}", title, msg);
}

fn run_command(
    program: &Path,
    args: &[String],
    cwd: &Path,
    wait_for_exit: bool,
    hide_console: bool,
) -> Result<i32, std::io::Error> {
    let mut cmd = Command::new(program);
    cmd.args(args).current_dir(cwd);
    #[cfg(windows)]
    if hide_console && !wait_for_exit {
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    if wait_for_exit {
        let status = cmd.status()?;
        Ok(status.code().unwrap_or(1))
    } else {
        cmd.spawn()?;
        Ok(0)
    }
}

fn main() -> ExitCode {
    let raw_args: Vec<String> = env::args().skip(1).collect();
    let wait_mode = raw_args
        .iter()
        .any(|a| a.eq_ignore_ascii_case("--smoke-test") || a.eq_ignore_ascii_case("--wait"));

    let exe_dir: PathBuf = env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_else(|| env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    // Priority 1: Standalone PyInstaller bundle (dist\AirTeachAI\AirTeachAI.exe)
    let dist_exe = exe_dir.join("dist").join("AirTeachAI").join("AirTeachAI.exe");
    if dist_exe.exists() {
        let work_dir = dist_exe.parent().unwrap_or(&exe_dir);
        match run_command(&dist_exe, &raw_args, work_dir, wait_mode, false) {
            Ok(code) => return ExitCode::from((code & 0xFF) as u8),
            Err(_) => {}
        }
    }

    // Priority 2: Adjacent standalone AirTeachAI.exe
    let local_exe = exe_dir.join("AirTeachAI.exe");
    if local_exe.exists() {
        match run_command(&local_exe, &raw_args, &exe_dir, wait_mode, false) {
            Ok(code) => return ExitCode::from((code & 0xFF) as u8),
            Err(_) => {}
        }
    }

    // Priority 3: Local virtual environment (.venv) launching main.py
    let main_py = exe_dir.join("main.py");
    if main_py.exists() {
        let py_candidates = if wait_mode {
            vec![
                exe_dir.join(".venv").join("Scripts").join("python.exe"),
                exe_dir.join(".venv").join("Scripts").join("pythonw.exe"),
            ]
        } else {
            vec![
                exe_dir.join(".venv").join("Scripts").join("pythonw.exe"),
                exe_dir.join(".venv").join("Scripts").join("python.exe"),
            ]
        };

        for py in py_candidates {
            if py.exists() {
                let mut full_args = Vec::with_capacity(raw_args.len() + 1);
                full_args.push(main_py.to_string_lossy().into_owned());
                full_args.extend(raw_args.iter().cloned());
                let hide = py
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.eq_ignore_ascii_case("python.exe"))
                    .unwrap_or(false);
                if let Ok(code) = run_command(&py, &full_args, &exe_dir, wait_mode, hide) {
                    return ExitCode::from((code & 0xFF) as u8);
                }
            }
        }

        // Priority 4: System Python in PATH
        let sys_bins = if wait_mode {
            ["python.exe", "pythonw.exe"]
        } else {
            ["pythonw.exe", "python.exe"]
        };
        for bin in sys_bins {
            let mut full_args = Vec::with_capacity(raw_args.len() + 1);
            full_args.push(main_py.to_string_lossy().into_owned());
            full_args.extend(raw_args.iter().cloned());
            if let Ok(code) = run_command(Path::new(bin), &full_args, &exe_dir, wait_mode, true) {
                return ExitCode::from((code & 0xFF) as u8);
            }
        }
    }

    show_error_dialog(
        "AirTeach AI — Chưa cài đặt môi trường",
        "Không tìm thấy dist\\AirTeachAI\\AirTeachAI.exe hoặc môi trường .venv.\n\nVui lòng nháy đúp vào \"setup_all.bat\" sau đó \"build_exe.bat\" (hoặc \"auto_setup_and_build.bat\") để cài đặt và đóng gói ứng dụng.",
    );
    ExitCode::from(1)
}
