"""
Patch STEM diagram methods into src/ui/stem_tools.py:
- PhysicsTools: create_harmonic_oscillator_diagram, create_radioactive_decay_diagram, create_gravitational_orbits_diagram
- ChemistryTools: create_reaction_kinetics_diagram
- InformaticsTools: create_rsa_crypto_diagram
"""

PHYSICS_METHODS = '''
    @staticmethod
    def create_harmonic_oscillator_diagram(
        center: tuple[float, float],
        width: float = 440.0,
        height: float = 260.0,
        mass_kg: float = 1.0,
        spring_k: float = 50.0,
        amplitude_px: float = 45.0,
        damping_b: float = 0.5,
        omega_drive: float = 7.07,
        mode: str = "SHM",
    ) -> list[QGraphicsItem]:
        """Creates laboratory harmonic oscillator and waveform vector diagram."""
        cx, cy = center
        items: list[QGraphicsItem] = []
        x_left = cx - width / 2.0
        y_top = cy - height / 2.0

        omega_0 = math.sqrt(max(0.01, spring_k / max(0.01, mass_kg)))
        banner_txt = f"DAO ĐỘNG CƠ HỌC: m = {mass_kg:.1f}kg, k = {spring_k:.0f}N/m | ω₀ = {omega_0:.2f} rad/s"
        banner = QGraphicsTextItem(banner_txt)
        banner.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        banner.setDefaultTextColor(QColor("#0f172a"))
        banner.setPos(x_left + 10.0, y_top - 24.0)
        banner.setZValue(13.0)
        items.append(banner)

        bg = QGraphicsRectItem(x_left, y_top, width, height)
        bg.setBrush(QBrush(QColor("#f8fafc")))
        bg.setPen(QPen(QColor("#cbd5e1"), 1.0))
        bg.setZValue(10.0)
        items.append(bg)

        wall_x = x_left + 20.0
        wall_y = y_top + 20.0
        wall_h = 70.0
        wall = QGraphicsLineItem(wall_x, wall_y, wall_x, wall_y + wall_h)
        wall.setPen(QPen(QColor("#475569"), 3.0))
        wall.setZValue(10.5)
        items.append(wall)

        mass_x = wall_x + 90.0 + amplitude_px * 0.4
        mass_y = wall_y + wall_h / 2.0
        spring_path = QPainterPath()
        spring_path.moveTo(wall_x, mass_y)
        coils = 8
        coil_len = (mass_x - wall_x - 15.0) / coils
        for i in range(coils):
            x_mid = wall_x + (i + 0.5) * coil_len
            y_disp = mass_y - 10.0 if i % 2 == 0 else mass_y + 10.0
            spring_path.lineTo(x_mid, y_disp)
        spring_path.lineTo(mass_x, mass_y)

        sp_item = QGraphicsPathItem(spring_path)
        sp_item.setPen(QPen(QColor("#0284c7"), 1.8))
        sp_item.setZValue(11.0)
        items.append(sp_item)

        mw, mh = 32.0, 32.0
        mass_block = QGraphicsRectItem(mass_x, mass_y - mh / 2.0, mw, mh)
        mass_block.setBrush(QBrush(QColor("#38bdf8")))
        mass_block.setPen(QPen(QColor("#0284c7"), 1.5))
        mass_block.setZValue(11.5)
        items.append(mass_block)

        m_lbl = QGraphicsTextItem("m")
        m_lbl.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        m_lbl.setDefaultTextColor(QColor("#0f172a"))
        m_lbl.setPos(mass_x + 8.0, mass_y - 12.0)
        m_lbl.setZValue(11.6)
        items.append(m_lbl)

        scope_x = x_left + 175.0
        scope_y = y_top + 20.0
        scope_w = width - 195.0
        scope_h = 130.0

        sc_bg = QGraphicsRectItem(scope_x, scope_y, scope_w, scope_h)
        sc_bg.setBrush(QBrush(QColor("#0f172a")))
        sc_bg.setPen(QPen(QColor("#334155"), 1.0))
        sc_bg.setZValue(10.5)
        items.append(sc_bg)

        mid_sc_y = scope_y + scope_h / 2.0
        sc_zero = QGraphicsLineItem(scope_x, mid_sc_y, scope_x + scope_w, mid_sc_y)
        sc_zero.setPen(QPen(QColor("#334155"), 1.0, Qt.PenStyle.DashLine))
        sc_zero.setZValue(10.6)
        items.append(sc_zero)

        wave_path = QPainterPath()
        pts = 60
        gamma = damping_b / (2.0 * max(0.01, mass_kg))
        for p in range(pts):
            t = p * 0.1
            px = scope_x + (p / float(pts)) * scope_w
            decay = math.exp(-gamma * t) if mode != "SHM" else 1.0
            disp = amplitude_px * 0.7 * decay * math.cos(omega_0 * t)
            py = mid_sc_y - disp
            if p == 0:
                wave_path.moveTo(px, py)
            else:
                wave_path.lineTo(px, py)

        w_item = QGraphicsPathItem(wave_path)
        w_item.setPen(QPen(QColor("#22c55e"), 1.8))
        w_item.setZValue(11.0)
        items.append(w_item)

        eq_y = y_top + height - 85.0
        eq_txt = (
            f"Phương Trình Vi Phân: m·x'' + b·x' + k·x = F₀·cos(ωt)\\n"
            f"Tần Số Riêng: ω₀ = √(k/m) = {omega_0:.2f} rad/s | Chu Kỳ T = 2π/ω₀ = {2.0*math.pi/omega_0:.2f}s\\n"
            f"Hệ Số Tắt Dần: γ = b/(2m) = {gamma:.2f} s⁻¹ | Cơ Năng: E = ½k·A² + ½m·v² = const (SHM)"
        )
        eq_item = QGraphicsTextItem(eq_txt)
        eq_item.setFont(QFont("Consolas", 8))
        eq_item.setDefaultTextColor(QColor("#334155"))
        eq_item.setPos(x_left + 15.0, eq_y)
        eq_item.setZValue(11.0)
        items.append(eq_item)

        return items

    @staticmethod
    def create_radioactive_decay_diagram(
        center: tuple[float, float],
        width: float = 440.0,
        height: float = 260.0,
        isotope: str = "CARBON_14",
        n_undecayed: int = 50,
        n_total: int = 100,
        elapsed_t_half: float = 1.0,
        half_life_str: str = "5730 năm",
    ) -> list[QGraphicsItem]:
        """Creates laboratory radioactive decay lattice and exponential curve vector diagram."""
        cx, cy = center
        items: list[QGraphicsItem] = []
        x_left = cx - width / 2.0
        y_top = cy - height / 2.0

        iso_specs = {
            "CARBON_14": ("Carbon-14 (¹⁴₆C)", "¹⁴₆C ➔ ¹⁴₇N + e⁻ + ν̄ₑ", "β⁻ (Beta)", "#f59e0b", "#10b981"),
            "RADON_222": ("Radon-222 (²²²₈₆Rn)", "²²²₈₆Rn ➔ ²¹⁸₈₄Po + ⁴₂He", "α (Alpha)", "#ef4444", "#06b6d4"),
            "IODINE_131": ("Iodine-131 (¹³¹₅₃I)", "¹³¹₅₃I ➔ ¹³¹₅₄Xe + e⁻ + γ", "β⁻, γ", "#a855f7", "#38bdf8"),
            "URANIUM_238": ("Uranium-238 (²³⁸₉₂U)", "²³⁸₉₂U ➔ ²³⁴₉₀Th + ⁴₂He", "α (Alpha)", "#eab308", "#10b981"),
            "COBALT_60": ("Cobalt-60 (⁶⁰₂₇Co)", "⁶⁰₂₇Co ➔ ⁶⁰₂₈Ni + e⁻ + γ", "β⁻, γ", "#3b82f6", "#22c55e"),
        }
        spec = iso_specs.get(isotope.upper(), iso_specs["CARBON_14"])
        iso_name, decay_eq, rad_type, col_parent, col_daughter = spec

        banner_txt = f"ĐỊNH LUẬT PHÓNG XẠ: {iso_name} | T½ = {half_life_str}"
        banner = QGraphicsTextItem(banner_txt)
        banner.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        banner.setDefaultTextColor(QColor("#0f172a"))
        banner.setPos(x_left + 10.0, y_top - 24.0)
        banner.setZValue(13.0)
        items.append(banner)

        bg = QGraphicsRectItem(x_left, y_top, width, height)
        bg.setBrush(QBrush(QColor("#f8fafc")))
        bg.setPen(QPen(QColor("#cbd5e1"), 1.0))
        bg.setZValue(10.0)
        items.append(bg)

        lat_x = x_left + 15.0
        lat_y = y_top + 20.0
        lat_w = 120.0
        lat_h = 120.0
        lat_bg = QGraphicsRectItem(lat_x, lat_y, lat_w, lat_h)
        lat_bg.setBrush(QBrush(QColor("#0f172a")))
        lat_bg.setPen(QPen(QColor("#334155"), 1.0))
        lat_bg.setZValue(10.5)
        items.append(lat_bg)

        grid_cols = 10
        dot_step = lat_w / grid_cols
        for idx in range(100):
            row = idx // grid_cols
            col = idx % grid_cols
            px = lat_x + col * dot_step + dot_step / 2.0
            py = lat_y + row * dot_step + dot_step / 2.0
            is_parent = idx < n_undecayed
            dcol = col_parent if is_parent else col_daughter
            dot = QGraphicsEllipseItem(px - 3.5, py - 3.5, 7.0, 7.0)
            dot.setBrush(QBrush(QColor(dcol)))
            dot.setPen(QPen(Qt.PenStyle.NoPen))
            dot.setZValue(11.0)
            items.append(dot)

        curve_x = x_left + 155.0
        curve_y = y_top + 20.0
        curve_w = width - 175.0
        curve_h = 120.0
        c_bg = QGraphicsRectItem(curve_x, curve_y, curve_w, curve_h)
        c_bg.setBrush(QBrush(QColor("#ffffff")))
        c_bg.setPen(QPen(QColor("#cbd5e1"), 1.0))
        c_bg.setZValue(10.5)
        items.append(c_bg)

        curve_path = QPainterPath()
        steps = 50
        for s in range(steps + 1):
            t_rel = (s / float(steps)) * 4.0
            px = curve_x + (s / float(steps)) * curve_w
            n_ratio = 2.0 ** (-t_rel)
            py = curve_y + curve_h - n_ratio * (curve_h - 15.0) - 8.0
            if s == 0:
                curve_path.moveTo(px, py)
            else:
                curve_path.lineTo(px, py)

        c_item = QGraphicsPathItem(curve_path)
        c_item.setPen(QPen(QColor("#ef4444"), 2.0))
        c_item.setZValue(11.0)
        items.append(c_item)

        cur_ratio = n_undecayed / float(max(1, n_total))
        marker_x = curve_x + min(1.0, elapsed_t_half / 4.0) * curve_w
        marker_y = curve_y + curve_h - cur_ratio * (curve_h - 15.0) - 8.0
        marker = QGraphicsEllipseItem(marker_x - 4.0, marker_y - 4.0, 8.0, 8.0)
        marker.setBrush(QBrush(QColor("#2563eb")))
        marker.setPen(QPen(QColor("#ffffff"), 1.5))
        marker.setZValue(11.5)
        items.append(marker)

        eq_y = y_top + height - 85.0
        eq_txt = (
            f"Định Luật: N(t) = N₀ · 2^(-t/T½) = N₀ · e^(-λt) | λ = ln(2)/T½\\n"
            f"Phản Ứng Phân Rã: {decay_eq} (Tia bức xạ: {rad_type})\\n"
            f"Hiện Tại: N/N₀ = {n_undecayed}/{n_total} ({cur_ratio*100:.1f}%) | Đã trôi qua: {elapsed_t_half:.2f} chu kỳ T½"
        )
        eq_item = QGraphicsTextItem(eq_txt)
        eq_item.setFont(QFont("Consolas", 8))
        eq_item.setDefaultTextColor(QColor("#334155"))
        eq_item.setPos(x_left + 15.0, eq_y)
        eq_item.setZValue(11.0)
        items.append(eq_item)

        return items

    @staticmethod
    def create_gravitational_orbits_diagram(
        center: tuple[float, float],
        width: float = 460.0,
        height: float = 270.0,
        preset: str = "INNER_SOLAR_SYSTEM",
        bodies_info: Optional[list[dict]] = None,
        show_lagrange: bool = False,
    ) -> list[QGraphicsItem]:
        """Creates astronomical & orbital mechanics vector diagram."""
        cx, cy = center
        items: list[QGraphicsItem] = []
        x_left = cx - width / 2.0
        y_top = cy - height / 2.0

        preset_titles = {
            "INNER_SOLAR_SYSTEM": ("Hệ Mặt Trời Cận Điểm (Inner Solar System)", "T² / a³ = 4π² / (G·M☉)"),
            "FIGURE_EIGHT_3BODY": ("Quỹ Đạo Hình Số 8 Đồng Thời (Figure-Eight 3-Body Choreography)", "Chenciner-Montgomery (2000)"),
            "EARTH_MOON_LAGRANGE": ("Hệ Trái Đất - Mặt Trăng & 5 Điểm Lagrange (L₁ - L₅)", "∇Φ_eff = 0 (Rotating Frame)"),
            "BINARY_STAR_PLANET": ("Sao Đôi & Hành Tinh Vòng Quanh (Circumbinary P-Type Orbit)", "Barycentric Keplerian Orbit"),
        }
        title_info = preset_titles.get(preset.upper(), preset_titles["INNER_SOLAR_SYSTEM"])
        preset_name, theory_sub = title_info

        banner_txt = f"VẠN VẬT HẤP DẪN & QUỸ ĐẠO: {preset_name}"
        banner = QGraphicsTextItem(banner_txt)
        banner.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        banner.setDefaultTextColor(QColor("#0f172a"))
        banner.setPos(x_left + 10.0, y_top - 24.0)
        banner.setZValue(13.0)
        items.append(banner)

        pane_x = x_left + 10.0
        pane_y = y_top + 10.0
        pane_w = 270.0
        pane_h = 230.0

        bg_pane = QGraphicsRectItem(pane_x, pane_y, pane_w, pane_h)
        bg_pane.setBrush(QBrush(QColor("#030712")))
        bg_pane.setPen(QPen(QColor("#1e293b"), 1.2))
        bg_pane.setZValue(10.0)
        items.append(bg_pane)

        orbit_center_x = pane_x + pane_w / 2.0
        orbit_center_y = pane_y + pane_h / 2.0

        if preset.upper() == "FIGURE_EIGHT_3BODY":
            path_8 = QPainterPath()
            s_scale = 80.0
            steps = 100
            for k in range(steps + 1):
                t = k * 2.0 * math.pi / steps
                den = 1.0 + math.sin(t) ** 2
                px = orbit_center_x + (s_scale * math.cos(t)) / den
                py = orbit_center_y + (s_scale * math.sin(t) * math.cos(t)) / den
                if k == 0:
                    path_8.moveTo(px, py)
                else:
                    path_8.lineTo(px, py)

            item_path = QGraphicsPathItem(path_8)
            item_path.setPen(QPen(QColor("#38bdf8"), 1.4, Qt.PenStyle.DashLine))
            item_path.setZValue(10.5)
            items.append(item_path)

            body_phases = [0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0]
            body_colors = ["#f59e0b", "#10b981", "#ec4899"]
            body_labels = ["m₁", "m₂", "m₃"]

            for bp, bc, bl in zip(body_phases, body_colors, body_labels):
                den = 1.0 + math.sin(bp) ** 2
                bx = orbit_center_x + (s_scale * math.cos(bp)) / den
                by = orbit_center_y + (s_scale * math.sin(bp) * math.cos(bp)) / den

                bdot = QGraphicsEllipseItem(bx - 5.0, by - 5.0, 10.0, 10.0)
                bdot.setBrush(QBrush(QColor(bc)))
                bdot.setPen(QPen(QColor("#ffffff"), 1.0))
                bdot.setZValue(11.0)
                items.append(bdot)

                lbl = QGraphicsTextItem(bl)
                lbl.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
                lbl.setDefaultTextColor(QColor("#f8fafc"))
                lbl.setPos(bx + 6.0, by - 12.0)
                lbl.setZValue(11.2)
                items.append(lbl)

        elif preset.upper() == "EARTH_MOON_LAGRANGE":
            earth_x = orbit_center_x - 30.0
            earth_y = orbit_center_y
            r_moon = 85.0
            moon_x = earth_x + r_moon
            moon_y = earth_y

            moon_orb = QGraphicsEllipseItem(earth_x - r_moon, earth_y - r_moon, 2 * r_moon, 2 * r_moon)
            moon_orb.setPen(QPen(QColor("#475569"), 1.0, Qt.PenStyle.DashLine))
            moon_orb.setZValue(10.4)
            items.append(moon_orb)

            edot = QGraphicsEllipseItem(earth_x - 9.0, earth_y - 9.0, 18.0, 18.0)
            edot.setBrush(QBrush(QColor("#0284c7")))
            edot.setPen(QPen(QColor("#ffffff"), 1.2))
            edot.setZValue(11.0)
            items.append(edot)

            elbl = QGraphicsTextItem("Earth (M₁)")
            elbl.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            elbl.setDefaultTextColor(QColor("#bae6fd"))
            elbl.setPos(earth_x - 24.0, earth_y + 10.0)
            elbl.setZValue(11.2)
            items.append(elbl)

            mdot = QGraphicsEllipseItem(moon_x - 4.5, moon_y - 4.5, 9.0, 9.0)
            mdot.setBrush(QBrush(QColor("#cbd5e1")))
            mdot.setPen(QPen(QColor("#ffffff"), 1.0))
            mdot.setZValue(11.0)
            items.append(mdot)

            l1_x = earth_x + r_moon * 0.84
            l1_y = earth_y
            l2_x = earth_x + r_moon * 1.16
            l2_y = earth_y
            l3_x = earth_x - r_moon * 1.01
            l3_y = earth_y
            l4_x = earth_x + r_moon * math.cos(math.pi / 3.0)
            l4_y = earth_y - r_moon * math.sin(math.pi / 3.0)
            l5_x = earth_x + r_moon * math.cos(-math.pi / 3.0)
            l5_y = earth_y - r_moon * math.sin(-math.pi / 3.0)

            lagrange_pts = [("L₁", l1_x, l1_y, "#f43f5e"),
                            ("L₂", l2_x, l2_y, "#fb7185"),
                            ("L₃", l3_x, l3_y, "#e879f9"),
                            ("L₄", l4_x, l4_y, "#facc15"),
                            ("L₅", l5_x, l5_y, "#facc15")]

            for lname, lx, ly, lcol in lagrange_pts:
                lp_dot = QGraphicsEllipseItem(lx - 3.5, ly - 3.5, 7.0, 7.0)
                lp_dot.setBrush(QBrush(QColor(lcol)))
                lp_dot.setPen(QPen(QColor("#ffffff"), 1.0))
                lp_dot.setZValue(11.5)
                items.append(lp_dot)

                l_txt = QGraphicsTextItem(lname)
                l_txt.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
                l_txt.setDefaultTextColor(QColor(lcol))
                l_txt.setPos(lx - 7.0, ly - 16.0)
                l_txt.setZValue(11.6)
                items.append(l_txt)

        else:
            sun_r = 14.0
            sun_dot = QGraphicsEllipseItem(orbit_center_x - sun_r, orbit_center_y - sun_r, sun_r * 2, sun_r * 2)
            sun_dot.setBrush(QBrush(QColor("#fbbf24")))
            sun_dot.setPen(QPen(QColor("#ffffff"), 1.2))
            sun_dot.setZValue(11.0)
            items.append(sun_dot)

            slbl = QGraphicsTextItem("Sun (M☉)")
            slbl.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            slbl.setDefaultTextColor(QColor("#fef08a"))
            slbl.setPos(orbit_center_x - 20.0, orbit_center_y + sun_r + 2.0)
            slbl.setZValue(11.2)
            items.append(slbl)

            planets = [
                ("Mercury", 35.0, 0.25 * math.pi, 3.5, "#94a3b8"),
                ("Venus", 58.0, 0.85 * math.pi, 5.0, "#eab308"),
                ("Earth", 85.0, 1.45 * math.pi, 5.5, "#38bdf8"),
                ("Mars", 112.0, 0.10 * math.pi, 4.2, "#ef4444"),
            ]

            for pname, porb, pth, prad, pcol in planets:
                oring = QGraphicsEllipseItem(orbit_center_x - porb, orbit_center_y - porb, 2 * porb, 2 * porb)
                oring.setPen(QPen(QColor("#334155"), 1.0, Qt.PenStyle.DashLine))
                oring.setZValue(10.4)
                items.append(oring)

                px = orbit_center_x + porb * math.cos(pth)
                py = orbit_center_y + porb * math.sin(pth)

                pdot = QGraphicsEllipseItem(px - prad, py - prad, 2 * prad, 2 * prad)
                pdot.setBrush(QBrush(QColor(pcol)))
                pdot.setPen(QPen(QColor("#ffffff"), 0.8))
                pdot.setZValue(11.0)
                items.append(pdot)

                plbl = QGraphicsTextItem(pname)
                plbl.setFont(QFont("Segoe UI", 7))
                plbl.setDefaultTextColor(QColor(pcol))
                plbl.setPos(px + 4.0, py - 12.0)
                plbl.setZValue(11.2)
                items.append(plbl)

        info_x = pane_x + pane_w + 12.0
        info_y = pane_y
        info_w = width - (info_x - x_left) - 10.0

        card_bg = QGraphicsRectItem(info_x, info_y, info_w, pane_h)
        card_bg.setBrush(QBrush(QColor("#f8fafc")))
        card_bg.setPen(QPen(QColor("#cbd5e1"), 1.0))
        card_bg.setZValue(10.0)
        items.append(card_bg)

        card_header = QGraphicsRectItem(info_x, info_y, info_w, 24.0)
        card_header.setBrush(QBrush(QColor("#0284c7")))
        card_header.setPen(Qt.PenStyle.NoPen)
        card_header.setZValue(10.1)
        items.append(card_header)

        header_title = QGraphicsTextItem("CƠ HỌC THIÊN THỂ")
        header_title.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        header_title.setDefaultTextColor(QColor("#ffffff"))
        header_title.setPos(info_x + 6.0, info_y + 2.0)
        header_title.setZValue(10.2)
        items.append(header_title)

        equations = [
            ("Lực Hấp Dẫn Newton:", "F = G · (m₁ · m₂) / r²"),
            ("Bảo Toàn Cơ Năng:", "E = ½mv² - GMm/r = const"),
            ("Định Luật Kepler III:", "T² / a³ = 4π² / (G·M)"),
            ("Vận Tốc Vũ Trụ I:", "v₁ = √(G·M / r)"),
            ("Vận Tốc Vũ Trụ II:", "v₂ = √(2G·M / r) = v₁√2"),
            ("Tích Phân Symplectic:", "Velocity Verlet (O(Δt²))"),
        ]

        curr_y = info_y + 30.0
        for eq_label, eq_val in equations:
            lbl_item = QGraphicsTextItem(eq_label)
            lbl_item.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            lbl_item.setDefaultTextColor(QColor("#0369a1"))
            lbl_item.setPos(info_x + 6.0, curr_y)
            lbl_item.setZValue(10.2)
            items.append(lbl_item)

            val_item = QGraphicsTextItem(eq_val)
            val_item.setFont(QFont("Consolas", 7))
            val_item.setDefaultTextColor(QColor("#0f172a"))
            val_item.setPos(info_x + 6.0, curr_y + 12.0)
            val_item.setZValue(10.2)
            items.append(val_item)

            curr_y += 28.0

        return items
'''

CHEMISTRY_METHODS = '''
    @staticmethod
    def create_reaction_kinetics_diagram(
        center: tuple[float, float],
        width: float = 440.0,
        height: float = 260.0,
        reaction_name: str = "SYNTHESIS_AB",
        temperature_k: float = 300.0,
        has_catalyst: bool = False,
        ea_kj: float = 50.0,
        delta_h_kj: float = -20.0,
        conc_reactants: float = 1.0,
        conc_products: float = 0.5,
    ) -> list[QGraphicsItem]:
        """Creates reaction kinetics energy profile and concentration curve vector diagram."""
        cx, cy = center
        items: list[QGraphicsItem] = []
        x_left = cx - width / 2.0
        y_top = cy - height / 2.0

        banner_txt = f"ĐỘNG HÓA HỌC: {reaction_name} | T = {temperature_k:.0f}K | Ea = {ea_kj:.1f} kJ/mol"
        banner = QGraphicsTextItem(banner_txt)
        banner.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        banner.setDefaultTextColor(QColor("#0f172a"))
        banner.setPos(x_left + 10.0, y_top - 24.0)
        banner.setZValue(13.0)
        items.append(banner)

        bg = QGraphicsRectItem(x_left, y_top, width, height)
        bg.setBrush(QBrush(QColor("#f8fafc")))
        bg.setPen(QPen(QColor("#cbd5e1"), 1.0))
        bg.setZValue(10.0)
        items.append(bg)

        # Left: Potential Energy Profile
        plot_x = x_left + 15.0
        plot_y = y_top + 20.0
        plot_w = 170.0
        plot_h = 130.0

        p_bg = QGraphicsRectItem(plot_x, plot_y, plot_w, plot_h)
        p_bg.setBrush(QBrush(QColor("#0f172a")))
        p_bg.setPen(QPen(QColor("#334155"), 1.0))
        p_bg.setZValue(10.5)
        items.append(p_bg)

        # Energy barrier curve
        path_e = QPainterPath()
        path_e.moveTo(plot_x + 10.0, plot_y + 80.0)  # Reactants
        path_e.cubicTo(
            plot_x + 45.0, plot_y + 80.0 - ea_kj * 0.8,
            plot_x + 100.0, plot_y + 80.0 - ea_kj * 0.8,
            plot_x + 155.0, plot_y + 80.0 - delta_h_kj * 0.5  # Products
        )
        pe_item = QGraphicsPathItem(path_e)
        pe_item.setPen(QPen(QColor("#f59e0b"), 2.0))
        pe_item.setZValue(11.0)
        items.append(pe_item)

        # Right: Concentration curves [A](t) and [C](t)
        c_x = x_left + 200.0
        c_y = y_top + 20.0
        c_w = width - 215.0
        c_h = 130.0

        c_bg = QGraphicsRectItem(c_x, c_y, c_w, c_h)
        c_bg.setBrush(QBrush(QColor("#ffffff")))
        c_bg.setPen(QPen(QColor("#cbd5e1"), 1.0))
        c_bg.setZValue(10.5)
        items.append(c_bg)

        # Reactant decay [A]
        path_a = QPainterPath()
        # Product growth [C]
        path_c = QPainterPath()
        steps = 40
        for s in range(steps + 1):
            px = c_x + (s / float(steps)) * c_w
            decay = math.exp(-0.08 * s)
            py_a = c_y + 15.0 + (1.0 - decay) * (c_h - 30.0)
            py_c = c_y + c_h - 15.0 - (1.0 - decay) * (c_h - 30.0)
            if s == 0:
                path_a.moveTo(px, py_a)
                path_c.moveTo(px, py_c)
            else:
                path_a.lineTo(px, py_a)
                path_c.lineTo(px, py_c)

        ia = QGraphicsPathItem(path_a)
        ia.setPen(QPen(QColor("#38bdf8"), 1.8))
        ia.setZValue(11.0)
        items.append(ia)

        ic = QGraphicsPathItem(path_c)
        ic.setPen(QPen(QColor("#10b981"), 1.8))
        ic.setZValue(11.0)
        items.append(ic)

        # Equations note
        eq_y = y_top + height - 85.0
        eq_txt = (
            f"Phương Trình Arrhenius: k(T) = A · e^(-Ea / RT) | R = 8.314 J/(mol·K)\\n"
            f"Tốc Độ Phản Ứng: v = k_f·[A][B] - k_r·[C][D] | Cân Bằng: Kc = k_f / k_r\\n"
            f"Trạng Thái: Xúc tác = {'Có' if has_catalyst else 'Không'} | ΔH = {delta_h_kj:.1f} kJ/mol | T = {temperature_k:.0f}K"
        )
        eq_item = QGraphicsTextItem(eq_txt)
        eq_item.setFont(QFont("Consolas", 8))
        eq_item.setDefaultTextColor(QColor("#334155"))
        eq_item.setPos(x_left + 15.0, eq_y)
        eq_item.setZValue(11.0)
        items.append(eq_item)

        return items
'''

INFORMATICS_METHODS = '''
    @staticmethod
    def create_rsa_crypto_diagram(
        center: tuple[float, float],
        width: float = 440.0,
        height: float = 260.0,
        prime_p: int = 7,
        prime_q: int = 11,
        public_e: int = 7,
        private_d: int = 43,
        plaintext_m: int = 9,
        ciphertext_c: int = 37,
    ) -> list[QGraphicsItem]:
        """Creates RSA public-key cryptographic workflow and modular arithmetic circle diagram."""
        cx, cy = center
        items: list[QGraphicsItem] = []
        x_left = cx - width / 2.0
        y_top = cy - height / 2.0

        n_val = prime_p * prime_q
        phi_val = (prime_p - 1) * (prime_q - 1)

        banner_txt = f"MẬT MÃ HÓA RSA: p={prime_p}, q={prime_q} ➔ n={n_val}, φ(n)={phi_val}"
        banner = QGraphicsTextItem(banner_txt)
        banner.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        banner.setDefaultTextColor(QColor("#0f172a"))
        banner.setPos(x_left + 10.0, y_top - 24.0)
        banner.setZValue(13.0)
        items.append(banner)

        bg = QGraphicsRectItem(x_left, y_top, width, height)
        bg.setBrush(QBrush(QColor("#f8fafc")))
        bg.setPen(QPen(QColor("#cbd5e1"), 1.0))
        bg.setZValue(10.0)
        items.append(bg)

        # Left: Modular Arithmetic Circle Dial
        dial_cx = x_left + 80.0
        dial_cy = y_top + 80.0
        dial_r = 55.0

        d_bg = QGraphicsEllipseItem(dial_cx - dial_r, dial_cy - dial_r, dial_r * 2, dial_r * 2)
        d_bg.setBrush(QBrush(QColor("#0f172a")))
        d_bg.setPen(QPen(QColor("#38bdf8"), 1.5))
        d_bg.setZValue(10.5)
        items.append(d_bg)

        # Chords forming Cardioid (k=2)
        steps = 24
        for i in range(steps):
            j = (2 * i) % steps
            a1 = (i / float(steps)) * 2.0 * math.pi - math.pi / 2.0
            a2 = (j / float(steps)) * 2.0 * math.pi - math.pi / 2.0
            x1 = dial_cx + dial_r * math.cos(a1)
            y1 = dial_cy + dial_r * math.sin(a1)
            x2 = dial_cx + dial_r * math.cos(a2)
            y2 = dial_cy + dial_r * math.sin(a2)
            line = QGraphicsLineItem(x1, y1, x2, y2)
            line.setPen(QPen(QColor("#a855f7"), 0.8))
            line.setZValue(10.8)
            items.append(line)

        # Center dial label
        lbl_mod = QGraphicsTextItem(f"mod {n_val}")
        lbl_mod.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        lbl_mod.setDefaultTextColor(QColor("#f8fafc"))
        lbl_mod.setPos(dial_cx - 20.0, dial_cy - 8.0)
        lbl_mod.setZValue(11.0)
        items.append(lbl_mod)

        # Right: Encryption / Decryption Pipeline Cards
        pipe_x = x_left + 160.0
        pipe_y = y_top + 15.0
        pipe_w = width - 175.0
        pipe_h = 135.0

        p_bg = QGraphicsRectItem(pipe_x, pipe_y, pipe_w, pipe_h)
        p_bg.setBrush(QBrush(QColor("#ffffff")))
        p_bg.setPen(QPen(QColor("#cbd5e1"), 1.0))
        p_bg.setZValue(10.5)
        items.append(p_bg)

        cards = [
            ("Khóa Công Khai (Public):", f"(e={public_e}, n={n_val})", "#0284c7"),
            ("Khóa Bí Mật (Private):", f"(d={private_d}, n={n_val})", "#ef4444"),
            ("Mã Hóa (Encryption):", f"C = M^e mod n = {plaintext_m}^{public_e} mod {n_val} = {ciphertext_c}", "#16a34a"),
            ("Giải Mã (Decryption):", f"M = C^d mod n = {ciphertext_c}^{private_d} mod {n_val} = {plaintext_m}", "#7c3aed"),
        ]

        curr_y = pipe_y + 8.0
        for title, val_str, col_hex in cards:
            t_item = QGraphicsTextItem(title)
            t_item.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            t_item.setDefaultTextColor(QColor(col_hex))
            t_item.setPos(pipe_x + 8.0, curr_y)
            t_item.setZValue(11.0)
            items.append(t_item)

            v_item = QGraphicsTextItem(val_str)
            v_item.setFont(QFont("Consolas", 7))
            v_item.setDefaultTextColor(QColor("#0f172a"))
            v_item.setPos(pipe_x + 8.0, curr_y + 12.0)
            v_item.setZValue(11.0)
            items.append(v_item)

            curr_y += 28.0

        # Equations note
        eq_y = y_top + height - 85.0
        eq_txt = (
            f"Định Lý Euler: a^φ(n) ≡ 1 (mod n) với gcd(a, n) = 1 | e · d ≡ 1 (mod φ(n))\\n"
            f"Thuật Toán Euclid Mở Rộng: d = e⁻¹ mod φ(n) = {private_d}\\n"
            f"Bản Gốc M = {plaintext_m} ➔ Bản Mã C = {ciphertext_c} ➔ Giải Mã Thành Công: M = {plaintext_m}"
        )
        eq_item = QGraphicsTextItem(eq_txt)
        eq_item.setFont(QFont("Consolas", 8))
        eq_item.setDefaultTextColor(QColor("#334155"))
        eq_item.setPos(x_left + 15.0, eq_y)
        eq_item.setZValue(11.0)
        items.append(eq_item)

        return items
'''


def apply_patch():
    with open('src/ui/stem_tools.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. PhysicsTools patch
    target_chem = '# ─── Vector Tools for Chemistry ───────────────────────────────────────────────'
    idx_chem = content.find(target_chem)
    assert idx_chem != -1, "Target chemistry section marker not found!"
    content = content[:idx_chem] + PHYSICS_METHODS + '\n\n' + content[idx_chem:]

    # 2. ChemistryTools patch
    target_bio = '# ─── Vector Tools for Biology ─────────────────────────────────────────────────'
    idx_bio = content.find(target_bio)
    assert idx_bio != -1, "Target biology section marker not found!"
    content = content[:idx_bio] + CHEMISTRY_METHODS + '\n\n' + content[idx_bio:]

    # 3. InformaticsTools patch
    target_dialog = '# ─── STEM Subject Dialog for Interactive Tool Parameter Entry ─────────────────'
    idx_dialog = content.find(target_dialog)
    assert idx_dialog != -1, "Target STEM dialog marker not found!"
    content = content[:idx_dialog] + INFORMATICS_METHODS + '\n\n' + content[idx_dialog:]

    with open('src/ui/stem_tools.py', 'w', encoding='utf-8') as f:
        f.write(content)

    print("Successfully patched stem_tools.py!")


if __name__ == '__main__':
    apply_patch()
