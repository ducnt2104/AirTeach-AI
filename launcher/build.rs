use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn find_rc_exe() -> Option<PathBuf> {
    if let Ok(status) = Command::new("rc.exe").arg("/?").output() {
        if status.status.success() || !status.stdout.is_empty() || !status.stderr.is_empty() {
            return Some(PathBuf::from("rc.exe"));
        }
    }
    let kits_root = Path::new(r"C:\Program Files (x86)\Windows Kits\10\bin");
    if kits_root.exists() {
        if let Ok(entries) = fs::read_dir(kits_root) {
            let mut versions: Vec<PathBuf> = entries
                .filter_map(|e| e.ok().map(|x| x.path()))
                .filter(|p| p.is_dir())
                .collect();
            versions.sort();
            versions.reverse();
            for ver in versions {
                let cand = ver.join("x64").join("rc.exe");
                if cand.exists() {
                    return Some(cand);
                }
            }
        }
    }
    None
}

fn main() {
    println!("cargo:rerun-if-changed=../assets/icons/airteach.ico");
    println!("cargo:rerun-if-changed=../assets/logo.ico");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap_or_default());
    let primary_ico = manifest_dir.join("..").join("assets").join("icons").join("airteach.ico");
    let fallback_ico = manifest_dir.join("..").join("assets").join("logo.ico");
    let ico = if primary_ico.exists() {
        primary_ico
    } else if fallback_ico.exists() {
        fallback_ico
    } else {
        return;
    };
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap_or_default());
    let rc_file = out_dir.join("airteach_icon.rc");
    let res_file = out_dir.join("airteach_icon.res");
    let ico_escaped = ico.to_string_lossy().replace('\\', "\\\\");
    let rc_content = format!("1 ICON \"{}\"\n", ico_escaped);
    if fs::write(&rc_file, rc_content).is_err() {
        return;
    }
    if let Some(rc_exe) = find_rc_exe() {
        let ok = Command::new(rc_exe)
            .args([
                "/nologo",
                "/fo",
                &res_file.to_string_lossy(),
                &rc_file.to_string_lossy(),
            ])
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if ok && res_file.exists() {
            println!("cargo:rustc-link-arg-bins={}", res_file.display());
        }
    }
}
