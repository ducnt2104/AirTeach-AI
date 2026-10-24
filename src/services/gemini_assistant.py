"""

AirTeach AI - Gemini AI Assistant Service

==========================================

Provides AI-powered teaching assistance via Google Gemini API.

Features per-teacher API key and configuration isolation,

contextual lesson chat, multimodal image analysis, lesson planning,

quiz generation, document summarization, and offline fallback.

Offline fallback: local knowledge base when API is unavailable.

Zero secret exposure: all outputs pass through credential sanitization.

"""

from __future__ import annotations

import logging

import time

from typing import Any, Callable, Generator, List, Optional

from src.services.credential_service import TeacherAISettings, TeacherCredentialService

logger = logging.getLogger(__name__)

class CallableBool:

    """Boolean value that is also callable as a zero-argument function for UI compatibility."""

    def __init__(self, value: bool) -> None:

        self._val = bool(value)

    def __bool__(self) -> bool:

        return self._val

    def __call__(self) -> bool:

        return self._val

    def __eq__(self, other: object) -> bool:

        return self._val == bool(other)

    def __repr__(self) -> str:

        return str(self._val)

# ── Local offline knowledge base ──────────────────────────────────────────────

_OFFLINE_GUIDE: dict[str, str] = {

    # ── Điều khiển cử chỉ & Tính năng AirTeach AI ─────────────────────────────

    "point": (

        "Cử chỉ POINT (Laser Pointer): Duỗi thẳng ngón trỏ, ngón cái giơ rộng ra xa. "

        "Con trỏ laser màu đỏ sẽ xuất hiện trên màn hình chiếu."

    ),

    "draw": (

        "Cử chỉ DRAW (Vẽ/Viết): Duỗi ngón trỏ để vẽ nét tự nhiên (Chế độ Air Pen), "

        "hoặc chụm ngón trỏ với ngón cái (Chế độ Precision Pinch)."

    ),

    "pause": (

        "Cử chỉ PAUSE (Nhấc bút): Nắm tay lại (fist). Phần mềm sẽ kết thúc nét vẽ "

        "và tự động làm mịn hình dạng."

    ),

    "select": (

        "Cử chỉ SELECT (Chọn vùng): Giơ đồng thời ngón trỏ và ngón giữa, "

        "ngón áp út và ngón út khép lại. Kéo để khoanh vùng nội dung muốn chọn."

    ),

    "swipe": (

        "Cử chỉ SWIPE (Lật trang): Mở toàn bộ 5 ngón tay, vẫy nhanh sang trái "

        "hoặc phải trong vòng 250ms. Chờ 800ms giữa hai lần lật để tránh lật liên tiếp."

    ),

    "confirm": (

        "Cử chỉ CONFIRM (👍 Xác nhận): Giơ ngón cái thẳng đứng, giữ ít nhất 300ms. "

        "Dùng để xác nhận hành động (lưu, đăng nhập, nắn hình, v.v.)"

    ),

    "calibrate": (

        "Hiệu chỉnh camera: Vào menu Cài đặt > Hiệu chỉnh cử chỉ hoặc Hiệu chỉnh camera. "

        "Giữ tay tại các góc hoặc thực hiện cử chỉ theo hướng dẫn của Calibration Wizard."

    ),

    "shape": (

        "Công cụ làm nét tự động (Shape Snapping): Sau khi vẽ xong (cử chỉ PAUSE), "

        "hệ thống phân tích hình dạng và tự động nắn thành: Đường thẳng, Mũi tên, "

        "Hình tròn/Elip, Hình chữ nhật, Tam giác."

    ),

    "ocr": (

        "Nhận diện chữ viết tay: Khoanh vùng nét chữ bằng cử chỉ SELECT, "

        "xác nhận bằng 👍 hoặc nút Review. Hệ thống nhận diện và chuyển thành văn bản/công thức LaTeX."

    ),

    "airteach": (

        "Tệp .airteach: Định dạng gói bài giảng của AirTeach AI gồm ảnh slide, "

        "nét vẽ vector, văn bản nhận diện và liên kết. Dùng menu Tệp > Lưu bài giảng."

    ),

    # ── Môn Toán học (Mathematics) ────────────────────────────────────────────
    "pythagoras": (
        "Định lý Pythagoras (Pitago): Trong tam giác vuông có hai cạnh góc vuông là a, b và cạnh huyền là c:\n"
        "  a^2 + b^2 = c^2 (hay c² = a² + b², c = √(a² + b²)):\n"
        "Các bộ ba số Pythagoras thông dụng: (3, 4, 5), (5, 12, 13), (6, 8, 10), (7, 24, 25)."
    ),
    "pitago": (
        "Định lý Pythagoras (Pitago): Trong tam giác vuông có hai cạnh góc vuông là a, b và cạnh huyền là c:\n"
        "  a^2 + b^2 = c^2 (hay c² = a² + b², c = √(a² + b²)):\n"
        "Các bộ ba số Pythagoras thông dụng: (3, 4, 5), (5, 12, 13), (6, 8, 10), (7, 24, 25)."
    ),
    "pytago": (
        "Định lý Pythagoras (Pitago): Trong tam giác vuông có hai cạnh góc vuông là a, b và cạnh huyền là c:\n"
        "  a^2 + b^2 = c^2 (hay c² = a² + b², c = √(a² + b²)):\n"
        "Các bộ ba số Pythagoras thông dụng: (3, 4, 5), (5, 12, 13), (6, 8, 10), (7, 24, 25)."
    ),
    "dao ham": (

        "Công thức Đạo hàm cơ bản (Toán học):\n"

        "- (xⁿ)' = n·xⁿ⁻¹\n"

        "- (sin x)' = cos x;  (cos x)' = -sin x\n"

        "- (tan x)' = 1 / cos² x = 1 + tan² x\n"

        "- (eˣ)' = eˣ;  (aˣ)' = aˣ·ln(a)\n"

        "- (ln x)' = 1 / x;  (logₐ x)' = 1 / (x·ln a)\n"

        "- Đạo hàm tích: (u·v)' = u'v + uv'; Đạo hàm thương: (u/v)' = (u'v - uv') / v²."

    ),

    "tich phan": (

        "Công thức Tích phân cơ bản (Toán học):\n"

        "- Định lý Newton-Leibniz: ∫[a..b] f(x)dx = F(b) - F(a)\n"

        "- Tích phân từng phần: ∫ u dv = u·v - ∫ v du (quy tắc đặt u: Nhất Lô, Nhì Đa, Tam Lượng, Tứ Mũ)\n"

        "- ∫ xⁿ dx = xⁿ⁺¹/(n+1) + C (n ≠ -1);  ∫ (1/x) dx = ln|x| + C\n"

        "- ∫ sin x dx = -cos x + C;  ∫ cos x dx = sin x + C;  ∫ eˣ dx = eˣ + C."

    ),

    "luong giac": (

        "Hệ thức Lượng giác trọng tâm:\n"

        "- sin² x + cos² x = 1;  1 + tan² x = 1/cos² x\n"

        "- Công thức cộng: sin(a ± b) = sin a·cos b ± cos a·sin b;  cos(a ± b) = cos a·cos b ∓ sin a·sin b\n"

        "- Nhân đôi: sin 2x = 2·sin x·cos x;  cos 2x = cos² x - sin² x = 2cos² x - 1 = 1 - 2sin² x."

    ),

    "taylor": (

        "Khai triển Chuỗi Taylor / Maclaurin:\n"

        "  f(x) = f(a) + f'(a)(x-a)/1! + f''(a)(x-a)²/2! + ... + f⁽ⁿ⁾(a)(x-a)ⁿ/n! + Rₙ(x)\n"

        "- Chuỗi Maclaurin eˣ = 1 + x + x²/2! + x³/3! + ...\n"

        "- Chuỗi Maclaurin sin x = x - x³/3! + x⁵/5! - ...;  cos x = 1 - x²/2! + x⁴/4! - ..."

    ),

    "euler": (

        "Đồng nhất thức Euler: e^(i·π) + 1 = 0, kết nối 5 hằng số toán học căn bản (e, i, π, 1, 0).\n"

        "Công thức Euler tổng quát: e^(i·θ) = cos(θ) + i·sin(θ)."

    ),

    "bac hai": (

        "Phương trình bậc hai ax² + bx + c = 0 (a ≠ 0):\n"

        "- Biệt thức Δ = b² - 4ac (hoặc Δ' = b'² - ac với b = 2b')\n"

        "- Δ > 0: Hai nghiệm phân biệt x₁,₂ = (-b ± √Δ) / (2a)\n"

        "- Δ = 0: Nghiệm kép x₁ = x₂ = -b / (2a)\n"

        "- Δ < 0: Phương trình vô nghiệm trên tập số thực (hai nghiệm phức trên ℂ)\n"

        "- Định lý Viète: S = x₁ + x₂ = -b/a;  P = x₁·x₂ = c/a."

    ),

    # ── Môn Vật lý (Physics) ──────────────────────────────────────────────────

    "newton": (

        "Ba Định luật Chuyển động của Newton:\n"

        "1. Định luật I (Quán tính): Một vật giữ nguyên trạng thái đứng yên hoặc chuyển động thẳng đều "

        "khi hợp lực tác dụng lên nó bằng 0 (∑F = 0 ⇒ a = 0).\n"

        "2. Định luật II: Gia tốc của một vật tỉ lệ thuận với hợp lực tác dụng và tỉ lệ nghịch với khối lượng: F = m·a.\n"

        "3. Định luật III (Tác dụng & Phản tác dụng): Khi vật A tác dụng lên vật B một lực, vật B cũng tác dụng "

        "lại vật A một lực cùng phương, ngược chiều và cùng độ lớn: F_AB = -F_BA."

    ),

    "ohm": (

        "Định luật Ohm & Năng lượng điện:\n"

        "- Định luật Ohm cho đoạn mạch: I = U / R  (Cường độ dòng điện I tỉ lệ thuận với HĐT U và tỉ lệ nghịch với điện trở R).\n"

        "- Định luật Joule-Lenz: Nhiệt lượng tỏa ra trên dây dẫn: Q = I²·R·t = U·I·t = (U²/R)·t (Joule).\n"

        "- Công suất điện: P = U·I = I²·R = U²/R (Watt)."

    ),

    "bohr": (

        "Hai Tiên đề của Bohr về nguyên tử Hydro:\n"

        "1. Tiên đề về các trạng thái dừng: Nguyên tử chỉ tồn tại trong những trạng thái có năng lượng xác định Eₙ, "

        "gọi là các trạng thái dừng. Khi ở trạng thái dừng, nguyên tử không bức xạ. Bán kính quỹ đạo dừng: rₙ = n²·r₀ (r₀ = 0.53 Å).\n"

        "2. Tiên đề về bức xạ và hấp thụ photon: Khi nguyên tử chuyển từ trạng thái dừng mức cao E_m sang mức thấp E_n, "

        "nó phát ra photon có năng lượng: ε = h·f = E_m - E_n."

    ),

    "compton": (

        "Tán xạ Compton (Hiệu ứng Compton - Lưỡng tính sóng-hạt của ánh sáng):\n"

        "Khi tán xạ trên electron tự do, photon tia X/gamma bị lệch góc θ và tăng bước sóng:\n"

        "  Δλ = λ' - λ = (h / (m_e·c)) · (1 - cos θ) = λ_C · (1 - cos θ)\n"

        "với bước sóng Compton λ_C ≈ 2.426 × 10⁻¹² m (0.02426 Å)."

    ),

    "doppler": (

        "Hiệu ứng Doppler:\n"

        "Tần số sóng f' mà máy thu nhận được thay đổi khi có chuyển động tương đối giữa nguồn phát và máy thu:\n"

        "  f' = f₀ · (v ± v_thu) / (v ∓ v_nguon)\n"

        "- Nguồn và máy thu lại gần nhau: f' > f₀ (âm sắc cao hơn, ánh sáng dịch chuyển về phía xanh).\n"

        "- Nguồn và máy thu ra xa nhau: f' < f₀ (âm sắc trầm hơn, ánh sáng dịch chuyển về phía đỏ)."

    ),

    "thau kinh": (

        "Công thức Thấu kính mỏng (Quang hình học):\n"

        "- Công thức độ tụ & tiêu cự: D = 1 / f = (n - 1)·(1/R₁ + 1/R₂)\n"

        "- Vị trí vật và ảnh: 1 / f = 1 / d + 1 / d'  (d: khoảng cách vật-thấu kính, d': khoảng cách ảnh-thấu kính)\n"

        "- Độ phóng đại ảnh: k = -d' / d = A'B' / AB (k > 0: ảnh cùng chiều; k < 0: ảnh ngược chiều)."

    ),

    "dao dong": (

        "Dao động điều hòa (Con lắc lò xo & con lắc đơn):\n"

        "- Phương trình li độ: x = A·cos(ωt + φ)\n"

        "- Vận tốc: v = x' = -ωA·sin(ωt + φ); Vận tốc cực đại v_max = ωA tại VTCB\n"

        "- Gia tốc: a = v' = -ω²x; Gia tốc cực đại a_max = ω²A tại biên\n"

        "- Con lắc lò xo: T = 2π√(m/k); Con lắc đơn: T = 2π√(l/g)."

    ),

    # ── Môn Hóa học (Chemistry) ───────────────────────────────────────────────

    "tuan hoan": (

        "Bảng tuần hoàn các nguyên tố hóa học Mendeleev:\n"

        "- Cấu trúc: 7 chu kỳ (chu kỳ nhỏ: 1, 2, 3; chu kỳ lớn: 4, 5, 6, 7) và 18 cột chia làm 8 nhóm A và 8 nhóm B.\n"

        "- Quy luật biến thiên trong 1 chu kỳ (trái → phải): Bán kính nguyên tử giảm, độ âm điện tăng, "

        "tính kim loại giảm, tính phi kim tăng.\n"

        "- Quy luật trong 1 nhóm A (trên → dưới): Bán kính nguyên tử tăng, độ âm điện giảm, "

        "tính kim loại tăng, tính phi kim giảm."

    ),

    "ph": (

        "Chỉ số pH & Môi trường dung dịch:\n"

        "- Định nghĩa: pH = -log[H⁺]  hay  [H⁺] = 10^(-pH)\n"

        "- Tích số ion của nước ở 25°C: K_w = [H⁺]·[OH⁻] = 1.0 × 10⁻¹⁴  ⇒  pH + pOH = 14\n"

        "- pH < 7: Môi trường axit ([H⁺] > [OH⁻])\n"

        "- pH = 7: Môi trường trung tính ([H⁺] = [OH⁻] = 10⁻⁷ M)\n"

        "- pH > 7: Môi trường bazơ/kiềm ([H⁺] < [OH⁻])."

    ),

    "can bang": (

        "Nguyên lý chuyển dịch cân bằng Le Chatelier:\n"

        "Một hệ phản ứng thuận nghịch đang ở trạng thái cân bằng, khi chịu một tác động từ bên ngoài "

        "(như thay đổi nhiệt độ, áp suất hoặc nồng độ), cân bằng sẽ tự động chuyển dịch theo chiều làm giảm tác động đó:\n"

        "- Tăng nồng độ chất phản ứng: Cân bằng chuyển dịch theo chiều thuận.\n"

        "- Tăng nhiệt độ: Cân bằng chuyển dịch theo chiều thu nhiệt (ΔH > 0).\n"

        "- Giảm áp suất: Cân bằng chuyển dịch theo chiều tăng số mol khí."

    ),

    "chuan do": (

        "Nguyên tắc Chuẩn độ Axit - Bazơ:\n"

        "Dựa trên phản ứng trung hòa: H⁺ + OH⁻ → H₂O.\n"

        "- Tại điểm tương đương: Số mol H⁺ = Số mol OH⁻  hay  C_A·V_A = C_B·V_B\n"

        "- Chất chỉ thị: Phenolphthalein (chuyển hồng ở pH ≈ 8.3), Quỳ tím (hóa đỏ khi axit, xanh khi kiềm)."

    ),

    "khi ly tuong": (

        "Phương trình trạng thái khí lý tưởng Clapeyron - Mendeleev:\n"

        "  P·V = n·R·T = (m / M)·R·T\n"

        "Trong đó: P là áp suất (Pa hoặc atm), V là thể tích (m³ hoặc lít), n là số mol,\n"

        "R = 8.314 J/(mol·K) hoặc 0.08206 L·atm/(mol·K), T là nhiệt độ tuyệt đối (Kelvin: T = t°C + 273.15)."

    ),

    # ── Môn Sinh học (Biology) ────────────────────────────────────────────────

    "adn": (

        "Cấu trúc & Nguyên tắc bổ sung của Phân tử ADN (DNA):\n"

        "- Cấu trúc xoắn kép: Gồm 2 chuỗi polynucleotide ngược chiều nhau (5'→3' và 3'→5') xoắn đều quanh một trục.\n"

        "- 4 loại đơn phân nucleotide: A (Adenine), T (Thymine), G (Guanine), X (Cytosine).\n"

        "- Nguyên tắc bổ sung giữa 2 mạch: A liên kết với T bằng 2 liên kết Hydro (A = T); "

        "G liên kết với X bằng 3 liên kết Hydro (G ≡ X).\n"

        "- Hệ quả: A = T, G = X; Tổng số nu: N = 2A + 2G = 2T + 2X; Chiều dài L = (N/2) × 3.4 Å."

    ),

    "dna": (

        "Cấu trúc & Nguyên tắc bổ sung của Phân tử ADN (DNA):\n"

        "- Cấu trúc xoắn kép: Gồm 2 chuỗi polynucleotide ngược chiều nhau (5'→3' và 3'→5') xoắn đều quanh một trục.\n"

        "- 4 loại đơn phân nucleotide: A (Adenine), T (Thymine), G (Guanine), X (Cytosine).\n"

        "- Nguyên tắc bổ sung giữa 2 mạch: A liên kết với T bằng 2 liên kết Hydro (A = T); "

        "G liên kết với X bằng 3 liên kết Hydro (G ≡ X).\n"

        "- Hệ quả: A = T, G = X; Tổng số nu: N = 2A + 2G = 2T + 2X; Chiều dài L = (N/2) × 3.4 Å."

    ),

    "menden": (

        "Quy luật di truyền Menđen:\n"

        "1. Quy luật Phân ly: Mỗi tính trạng do một cặp alen quy định. Trong quá trình giảm phân tạo giao tử, "

        "hai alen trong cặp phân ly đồng đều về các giao tử (F2 phân ly 3 trội : 1 lặn).\n"

        "2. Quy luật Phân ly độc lập: Các cặp nhân tố di truyền phân ly độc lập trong quá trình hình thành giao tử "

        "(Phép lai 2 cặp tính trạng F2 phân ly kiểu hình: 9:3:3:1)."

    ),

    "quang hop": (

        "Quá trình Quang hợp ở thực vật:\n"

        "- Phương trình tổng quát: 6CO₂ + 12H₂O ──(Ánh sáng, Diệp lục)──> C₆H₁₂O₆ + 6O₂ + 6H₂O\n"

        "- Pha sáng (xảy ra tại thylakoid): Sử dụng năng lượng ánh sáng để quang phân ly nước giải phóng O₂, "

        "tổng hợp năng lượng ATP và chất khử NADPH.\n"

        "- Pha tối / Chu trình Calvin (xảy ra tại stroma): Cố định CO₂ sử dụng ATP và NADPH từ pha sáng để tạo đường glucose."

    ),

    "ho hap": (

        "Hô hấp tế bào (Phân giải kị khí & hiếu khí):\n"

        "- Phương trình tổng quát hiếu khí: C₆H₁₂O₆ + 6O₂ → 6CO₂ + 6H₂O + ~30-32 ATP.\n"

        "- Ba giai đoạn chính:\n"

        "  1. Đường phân (Cytoplasm): 1 Glucose → 2 Pyruvate + 2 ATP + 2 NADH.\n"

        "  2. Chu trình Krebs (Ma trận ty thể): Tạo CO₂, 2 ATP, 6 NADH, 2 FADH₂.\n"

        "  3. Chuỗi chuyền electron hô hấp (Màng trong ty thể): Tổng hợp phần lớn ATP (~26-28 ATP)."

    ),

    # ── Môn Tin học (Computer Science & Algorithms) ───────────────────────────

    "nhi phan": (

        "Cây nhị phân & Cây tìm kiếm nhị phân (BST):\n"

        "- Mỗi nút có tối đa 2 con (con trái và con phải).\n"

        "- Tính chất BST: Mọi nút ở cây con trái có giá trị nhỏ hơn nút gốc, "

        "mọi nút ở cây con phải có giá trị lớn hơn nút gốc.\n"

        "- Các phép duyệt cây: Tiền thứ tự (NLR), Trung thứ tự (LNR - cho dãy tăng dần), Hậu thứ tự (LRN).\n"

        "- Độ phức tạp tìm kiếm/chèn/xóa trong BST cân bằng: O(log N); suy biến: O(N)."

    ),

    "sap xep": (

        "Độ phức tạp các thuật toán Sắp xếp (Sorting Algorithms):\n"

        "- QuickSort: Trung bình O(N log N), xấu nhất O(N²), bộ nhớ O(log N).\n"

        "- MergeSort: Luôn luôn O(N log N), sắp xếp ổn định (Stable), bộ nhớ O(N).\n"

        "- HeapSort: Luôn luôn O(N log N), không cần bộ nhớ phụ O(1).\n"

        "- Sắp xếp cơ bản (Bubble Sort, Insertion Sort, Selection Sort): O(N²)."

    ),

    "do thi": (

        "Thuật toán Đồ thị căn bản:\n"

        "- Duyệt đồ thị: DFS (Tìm kiếm theo chiều sâu - dùng Stack/đệ quy), BFS (Tìm kiếm theo chiều rộng - dùng Queue).\n"

        "- Cây khung nhỏ nhất (MST): Thuật toán Kruskal (dùng DSU/Union-Find, O(E log E)), Thuật toán Prim (dùng Min-Heap, O(E log V)).\n"

        "- Đường đi ngắn nhất nguồn đơn: Dijkstra (trọng số không âm, O((V + E) log V)), Bellman-Ford (chấp nhận trọng số âm, O(V·E))."

    ),

    # ── Trợ giúp mặc định ─────────────────────────────────────────────────────

    "default": (

        "AirTeach AI hỗ trợ các cử chỉ: POINT (laser), DRAW (viết), PAUSE (nhấc bút), "

        "SELECT (chọn vùng), SWIPE (lật trang), CONFIRM (👍).\n"

        "Đồng thời hỗ trợ tra cứu kiến thức STEM offline: 'pythagoras', 'đạo hàm', 'tích phân', "

        "'newton', 'ohm', 'bohr', 'tuần hoàn', 'ph', 'adn', 'quang hợp', 'sắp xếp', 'đồ thị'..."

    ),

}

def _normalize_query(s: str) -> str:

    """Normalize Vietnamese query to accent-free lowercase for robust keyword matching."""

    import unicodedata

    s = s.replace("đ", "d").replace("Đ", "d")

    return "".join(

        c for c in unicodedata.normalize("NFD", s.lower())

        if unicodedata.category(c) != "Mn"

    )

def _offline_lookup(query: str) -> str:

    """Find the best matching offline guide entry for *query*."""

    q_norm = _normalize_query(query)

    for keyword, answer in _OFFLINE_GUIDE.items():

        if _normalize_query(keyword) in q_norm:

            return answer

    return _OFFLINE_GUIDE["default"]

# ── Gemini Client Wrapper ─────────────────────────────────────────────────────

class GeminiAssistant:

    """

    Gemini API assistant with streaming support, per-teacher isolation,

    and offline fallback.

    Args:

        api_key:            Google AI Studio API key (optional if using credential_service).

        model_name:         Gemini model to use (default: gemini-2.5-flash).

        teacher_id:         Current active teacher profile ID.

        credential_service: TeacherCredentialService instance for per-teacher key management.

    """

    SYSTEM_PROMPT = (

        "Bạn là AirTeach AI Assistant — trợ lý giảng dạy thông minh tích hợp trong phần mềm "

        "AirTeach AI. Bạn hỗ trợ giáo viên Việt Nam giảng dạy hiệu quả hơn bằng cách:\n"

        "- Giải thích nội dung chuyên môn theo ngữ cảnh bài học đang mở.\n"

        "- Soạn giáo án, dàn ý bài giảng, tóm tắt bài học ngắn gọn và dễ hiểu.\n"

        "- Tạo câu hỏi trắc nghiệm, bài tập ôn luyện theo các mức độ nhận thức (Nhận biết, Thông hiểu, Vận dụng, Vận dụng cao).\n"

        "- Gợi ý cách dẫn dắt, ví dụ minh hoạ thực tế, câu hỏi tương tác trên lớp.\n"

        "- Hỗ trợ kiểm tra bài giải công thức toán học/vật lý/hoá học.\n"

        "- Giải đáp cách sử dụng phần mềm AirTeach AI.\n"

        "Luôn trả lời bằng tiếng Việt chuẩn mực, sư phạm, súc tích và thực tiễn."

    )

    def __init__(

        self,

        api_key: str = "",

        model_name: str = "gemini-2.5-flash",

        teacher_id: str = "",

        credential_service: Optional[TeacherCredentialService] = None,

    ) -> None:

        self._teacher_id = teacher_id

        self._cred_service = credential_service or TeacherCredentialService()

        self._model_name = model_name

        self._client = None

        self._online = False

        # Load key from credential service if teacher_id provided, else use passed api_key

        if self._teacher_id:

            stored_key = self._cred_service.get_api_key(self._teacher_id)

            settings = self._cred_service.get_settings(self._teacher_id)

            self._api_key = stored_key or api_key

            self._model_name = settings.model_name or model_name

        else:

            self._api_key = api_key

        self._init_client()

    def set_active_teacher(self, teacher_id: str) -> None:

        """

        Switch active teacher account: reloads that teacher's isolated API key & settings.

        Teacher A and Teacher B never share keys.

        """

        self._teacher_id = teacher_id

        self._client = None

        self._online = False

        if teacher_id:

            self._api_key = self._cred_service.get_api_key(teacher_id)

            settings = self._cred_service.get_settings(teacher_id)

            if settings and settings.model_name:

                self._model_name = settings.model_name

            logger.info("GeminiAssistant switched to teacher: %s (has_key=%s)", teacher_id, bool(self._api_key))

        else:

            self._api_key = ""

        self._init_client()

    def get_teacher_id(self) -> str:

        return self._teacher_id

    def _init_client(self) -> None:

        """Attempt to initialise the Gemini client with the current key."""

        if not self._api_key:

            logger.info("No Gemini API key for current teacher — running in offline pedagogical mode.")

            self._client = None

            self._online = False

            return

        try:

            from google import genai  # noqa: PLC0415

            self._client = genai.Client(api_key=self._api_key)

            self._online = True

            logger.info("Gemini client initialised for model=%s.", self._model_name)

        except Exception as exc:

            redacted_err = TeacherCredentialService.redact_text(str(exc))

            logger.warning("Gemini client init failed: %s — fallback to offline.", redacted_err)

            self._client = None

            self._online = False

    def update_api_key(self, new_key: str) -> bool:

        """

        Hot-update the API key for the current teacher.

        Saves key into secure OS Keyring under teacher namespace.

        """

        clean_key = new_key.strip()

        self._api_key = clean_key

        if self._teacher_id:

            self._cred_service.set_api_key(self._teacher_id, clean_key)

        self._client = None

        self._online = False

        self._init_client()

        return bool(self._online)

    def update_model_name(self, model_name: str) -> None:

        """Update active model (e.g. gemini-2.5-flash, gemini-1.5-pro)."""

        self._model_name = model_name

        if self._teacher_id:

            settings = self._cred_service.get_settings(self._teacher_id)

            settings.model_name = model_name

            self._cred_service.save_settings(settings)

        # Re-initialize client if necessary

        self._init_client()

    # ── Internal streaming helper ──────────────────────────────────────────────

    def _stream_generate(

        self,

        contents: list,

        system_instruction: str = "",

        on_chunk: Optional[Callable[[str], None]] = None,

    ) -> str:

        """

        Stream a Gemini generation request.

        Calls *on_chunk(text)* for each incremental text delta.

        Returns the full concatenated response.

        """

        from google.genai import types  # noqa: PLC0415

        config = types.GenerateContentConfig(

            system_instruction=system_instruction or self.SYSTEM_PROMPT,

            temperature=0.7,

            max_output_tokens=2048,

        )

        full_text = ""

        try:

            with self._client.models.generate_content_stream(

                model=self._model_name,

                contents=contents,

                config=config,

            ) as stream:

                for chunk in stream:

                    delta = chunk.text or ""

                    full_text += delta

                    if on_chunk and delta:

                        on_chunk(delta)

        except Exception as exc:

            redacted_err = TeacherCredentialService.redact_text(str(exc))

            logger.warning("Gemini streaming failed: %s", redacted_err)

            raise

        return full_text

    # ── Public API: Chat & Teaching Assistance ─────────────────────────────────

    def chat_with_teacher(

        self,

        history: list[dict],

        user_message: str,

        current_page_text: str = "",

        on_chunk: Optional[Callable[[str], None]] = None,

    ) -> str:

        """

        Contextual lesson chat.

        """

        if not self._online or self._client is None:

            fallback = (

                "Không thể kết nối Gemini API — đang ở chế độ ngoại tuyến.\n"

                "Vui lòng kiểm tra lại cấu hình API Key trong mục Cài đặt > AI & API.\n\n"

                + _offline_lookup(user_message)

            )

            if on_chunk:

                on_chunk(fallback)

            return fallback

        try:

            from google.genai import types  # noqa: PLC0415

            contents = []

            for turn in history:

                role = turn.get("role", "user")

                text = turn.get("text", "")

                contents.append(types.Content(

                    role=role,

                    parts=[types.Part(text=text)],

                ))

            context_prefix = ""

            if current_page_text.strip():

                context_prefix = (

                    f"[Nội dung bài giảng/slide đang mở]:\n{current_page_text[:2000]}\n\n"

                )

            contents.append(types.Content(

                role="user",

                parts=[types.Part(text=context_prefix + user_message)],

            ))

            return self._stream_generate(contents, on_chunk=on_chunk)

        except Exception as exc:

            redacted_err = TeacherCredentialService.redact_text(str(exc))

            err_msg = (

                f"Không thể xác thực API hoặc mất kết nối mạng. "

                f"Hệ thống tự động chuyển sang Trợ lý Hướng dẫn Offline nội bộ.\n\n"

                f"(Chi tiết: {redacted_err})\n\n"

                + _offline_lookup(user_message)

            )

            logger.warning("chat_with_teacher failed: %s", redacted_err)

            if on_chunk:

                on_chunk(err_msg)

            return err_msg

    def explain_canvas_region(

        self,

        image_bytes: bytes = b"",

        prompt: str = "Hãy giải thích nội dung trong hình này theo ngữ cảnh giảng dạy.",

        on_chunk: Optional[Callable[[str], None]] = None,

        cropped_image_bytes: Optional[bytes] = None,

    ) -> str:

        """

        Send a cropped canvas region image to Gemini Multimodal for pedagogical explanation.

        Supports both image_bytes and cropped_image_bytes keyword arguments.

        """

        actual_bytes = cropped_image_bytes if cropped_image_bytes is not None else image_bytes

        if not self._online or self._client is None:

            fallback = (

                "Chế độ ngoại tuyến: Không thể kết nối Gemini API để phân tích hình ảnh.\n"

                "Vui lòng thiết lập khóa API trong Cài đặt > AI & API."

            )

            if on_chunk:

                on_chunk(fallback)

            return fallback

        try:

            from google.genai import types  # noqa: PLC0415

            contents = [

                types.Content(

                    role="user",

                    parts=[

                        types.Part(

                            inline_data=types.Blob(

                                mime_type="image/png",

                                data=actual_bytes,

                            )

                        ),

                        types.Part(text=prompt),

                    ],

                )

            ]

            return self._stream_generate(contents, on_chunk=on_chunk)

        except Exception as exc:

            redacted_err = TeacherCredentialService.redact_text(str(exc))

            err_msg = (

                f"Lỗi khi gửi phân tích hình ảnh tới AI.\n"

                f"(Chi tiết: {redacted_err})"

            )

            logger.warning("explain_canvas_region failed: %s", redacted_err)

            if on_chunk:

                on_chunk(err_msg)

            return err_msg

    def get_interactive_usage_guide(

        self,

        query: str,

        on_chunk: Optional[Callable[[str], None]] = None,

    ) -> str:

        """

        Answer questions about AirTeach AI features, gestures, calibration.

        """

        if not self._online or self._client is None:

            answer = _offline_lookup(query)

            if on_chunk:

                on_chunk(answer)

            return answer

        guide_system = (

            self.SYSTEM_PROMPT + "\n\n"

            "Tập trung vào hướng dẫn sử dụng phần mềm AirTeach AI: "

            "cử chỉ tay, hiệu chỉnh camera, làm nét hình học, nhận diện chữ viết và công thức."

        )

        try:

            from google.genai import types  # noqa: PLC0415

            contents = [types.Content(role="user", parts=[types.Part(text=query)])]

            return self._stream_generate(contents, system_instruction=guide_system, on_chunk=on_chunk)

        except Exception as exc:

            answer = _offline_lookup(query)

            redacted_err = TeacherCredentialService.redact_text(str(exc))

            logger.warning("get_interactive_usage_guide failed: %s — fallback to offline.", redacted_err)

            if on_chunk:

                on_chunk(answer)

            return answer

    # ── AI Workbench Pedagogical Operations (Part VIII) ────────────────────────

    def generate_lesson_outline(

        self,

        subject: str,

        topic: str,

        grade_level: str = "THPT",

        on_chunk: Optional[Callable[[str], None]] = None,

    ) -> str:

        """Generate a structured lesson outline with objectives, steps, and key questions."""

        prompt = (

            f"Hãy lập dàn ý bài dạy chi tiết cho môn {subject}, bài học/chủ đề: '{topic}', "

            f"đối tượng học sinh: {grade_level}.\n\n"

            f"Dàn ý cần bao gồm:\n"

            f"1. Mục tiêu bài học (Kiến thức, Kỹ năng, Phẩm chất)\n"

            f"2. Khởi động / Dẫn dắt vào bài (3-5 phút)\n"

            f"3. Các hoạt động hình thành kiến thức trọng tâm\n"

            f"4. Hoạt động luyện tập & củng cố\n"

            f"5. Vận dụng thực tế & mở rộng"

        )

        return self._run_prompt_task(prompt, on_chunk=on_chunk)

    def summarize_document_content(

        self,

        text_content: str,

        max_points: int = 5,

        on_chunk: Optional[Callable[[str], None]] = None,

    ) -> str:

        """Summarize selected lesson document or slide content into concise bullet points."""

        prompt = (

            f"Hãy tóm tắt ngắn gọn nội dung văn bản sau thành tối đa {max_points} ý chính "

            f"thích hợp để giáo viên ghi bảng hoặc trình chiếu:\n\n"

            f"---\n{text_content[:4000]}\n---"

        )

        return self._run_prompt_task(prompt, on_chunk=on_chunk)

    def generate_quiz_questions(

        self,

        content: str,

        num_questions: int = 4,

        difficulty: str = "Đa cấp độ",

        on_chunk: Optional[Callable[[str], None]] = None,

    ) -> str:

        """Generate multiple choice or short-answer review questions with answer key."""

        prompt = (

            f"Dựa vào nội dung kiến thức sau, hãy soạn {num_questions} câu hỏi trắc nghiệm ôn tập "

            f"(độ khó: {difficulty}) kèm theo 4 phương án A, B, C, D, chỉ rõ đáp án đúng và lời giải thích ngắn gọn:\n\n"

            f"---\n{content[:4000]}\n---"

        )

        return self._run_prompt_task(prompt, on_chunk=on_chunk)

    def explain_concept(

        self,

        concept: str,

        target_audience: str = "Học sinh THPT",

        on_chunk: Optional[Callable[[str], None]] = None,

    ) -> str:

        """Explain a complex STEM concept with real-world analogies and visual cues."""

        prompt = (

            f"Hãy giải thích khái niệm '{concept}' một cách trực quan, dễ nhớ cho đối tượng {target_audience}.\n"

            f"Bao gồm: định nghĩa ngắn gọn, hình ảnh/ví dụ minh họa thực tế trong cuộc sống, "

            f"và một câu hỏi tương tác để giáo viên hỏi học sinh ngay trên lớp."

        )

        return self._run_prompt_task(prompt, on_chunk=on_chunk)

    def generate_lesson_plan(

        self,

        topic: str,

        duration_minutes: int = 45,

        on_chunk: Optional[Callable[[str], None]] = None,

    ) -> str:

        """Generate a complete 45-minute lesson plan according to pedagogical standards."""

        prompt = (

            f"Hãy soạn một kế hoạch bài dạy (giáo án) chuẩn thời lượng {duration_minutes} phút "

            f"cho chủ đề: '{topic}'. Phân bổ thời gian cụ thể cho từng hoạt động dạy học của giáo viên và học sinh."

        )

        return self._run_prompt_task(prompt, on_chunk=on_chunk)

    def _run_prompt_task(self, prompt: str, on_chunk: Optional[Callable[[str], None]] = None) -> str:

        """Internal runner for workbench generation tasks."""

        if not self._online or self._client is None:

            offline_ans = _offline_lookup(prompt)

            msg = (

                "⚠️ Đang ở chế độ ngoại tuyến: Không thể gọi Gemini API trực tuyến.\n"

                "Hệ thống tự động trích xuất kiến thức sư phạm từ Cơ sở Tri thức Offline:\n\n"

                + offline_ans

            )

            if on_chunk:

                on_chunk(msg)

            return msg

        try:

            from google.genai import types  # noqa: PLC0415

            contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]

            return self._stream_generate(contents, on_chunk=on_chunk)

        except Exception as exc:

            redacted_err = TeacherCredentialService.redact_text(str(exc))

            err_msg = f"Lỗi khi thực hiện yêu cầu AI: {redacted_err}"

            logger.warning("AI Workbench prompt task failed: %s", redacted_err)

            if on_chunk:

                on_chunk(err_msg)

            return err_msg

    # ── Status ─────────────────────────────────────────────────────────────────

    @property

    def is_online(self) -> CallableBool:

        """

        Boolean status whether the Gemini API is initialized and online.

        Compatible with both attribute access (if assistant.is_online)

        and method call (if assistant.is_online()).

        """

        return CallableBool(self._online)

    def model_name(self) -> str:

        return self._model_name
