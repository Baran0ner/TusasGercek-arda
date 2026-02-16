"""
Multi-Zone Laminate Optimizer

Birden fazla zone için uyumlu katman dizilimleri oluşturur.
Root zone optimize edilir, diğer zone'lar drop-off ile türetilir.

Fiziksel kısıtlar:
  - Ağırlık minimizasyonu: Weight = Σ (Area_i × PlyCount_i × Density)
  - Ramp kısıtı: Komşu zone'lar arası ply farkı için yeterli fiziksel mesafe gerekir
"""

import math
from collections import Counter
from typing import Dict, List, Tuple, Any, Optional

from .laminate_optimizer import LaminateOptimizer
from .dropoff_optimizer import DropOffOptimizer


# ===== Fiziksel sabitler =====
PLY_THICKNESS_MM = 0.125          # Tipik prepreg ply kalınlığı (mm)
CFRP_DENSITY_G_MM3 = 1.58e-3     # Karbon fiber takviyeli polimer yoğunluğu (g/mm³)
RAMP_RATE_MM_PER_PLY = 0.5       # Her ply düşüşü için gereken minimum ramp mesafesi (mm)
                                   # Endüstri standardı: ~0.3-0.6 mm/ply


class MultiZoneOptimizer:
    """
    Çoklu zone optimizasyonu için ana sınıf.
    
    Kullanım:
        zones_config = [
            {"0": 12, "90": 8, "45": 8, "-45": 8},   # Zone 1: 36 ply
            {"0": 8, "90": 8, "45": 8, "-45": 8},    # Zone 2: 32 ply
            {"0": 6, "90": 6, "45": 6, "-45": 6},    # Zone 3: 24 ply
        ]
        # bounds: her zone için {"x": px, "y": px, "w": px, "h": px} (piksel veya mm)
        optimizer = MultiZoneOptimizer(zones_config, bounds=bounds_list, panel_scale_mm=300)
        results = optimizer.optimize_all()
    """

    MAX_ROOT_RETRIES = 5  # Root güncellemesi için maksimum deneme

    def __init__(self, zones_config: List[Dict[int, int]],
                 bounds: Optional[List[Dict]] = None,
                 panel_scale_mm: float = 300.0):
        """
        Args:
            zones_config: Her zone için açı sayılarını içeren dict listesi.
            bounds: Her zone için {"x","y","w","h"} konumları (piksel; normalize edilir).
            panel_scale_mm: Panel'in en uzun kenarının mm cinsinden boyutu
                            (piksel -> mm dönüşümü için).
        """
        # String key'leri int'e çevir
        self.zones_config = []
        for zone in zones_config:
            converted = {int(k): int(v) for k, v in zone.items()}
            self.zones_config.append(converted)
        
        # Her zone'un toplam ply sayısını hesapla
        self.zone_totals = [sum(z.values()) for z in self.zones_config]
        
        # Root zone'u belirle (en kalın)
        self.root_index = self.zone_totals.index(max(self.zone_totals))
        
        # Zone'ları kalınlıktan inceliğe sırala (indekslerle birlikte)
        self.sorted_zone_indices = sorted(
            range(len(self.zones_config)),
            key=lambda i: self.zone_totals[i],
            reverse=True  # En kalından en inceye
        )

        # ===== Geometrik veriler =====
        self.bounds = bounds  # piksel cinsinden
        self.panel_scale_mm = panel_scale_mm
        self.zone_areas_mm2 = []    # Her zone'un gerçek alanı (mm²)
        self.zone_dims_mm = []      # Her zone'un (w_mm, h_mm) boyutları
        self.zone_neighbors = []    # Komşuluk listesi (adjacency)

        if bounds and len(bounds) == len(zones_config):
            self._compute_geometry(bounds, panel_scale_mm)

    def _compute_geometry(self, bounds: List[Dict], panel_scale_mm: float):
        """Piksel cinsinden bounds bilgisini mm cinsine çevir ve komşuluk hesapla."""
        # Tüm zone'ların kapsadığı alan bounding box
        all_x = [b["x"] for b in bounds] + [b["x"] + b["w"] for b in bounds]
        all_y = [b["y"] for b in bounds] + [b["y"] + b["h"] for b in bounds]
        max_extent_px = max(max(all_x) - min(all_x), max(all_y) - min(all_y), 1)
        
        # Piksel -> mm ölçek faktörü
        scale = panel_scale_mm / max_extent_px

        self.zone_areas_mm2 = []
        self.zone_dims_mm = []
        for b in bounds:
            w_mm = b["w"] * scale
            h_mm = b["h"] * scale
            self.zone_areas_mm2.append(w_mm * h_mm)
            self.zone_dims_mm.append((w_mm, h_mm))

        # Komşuluk: iki zone'un kenarları birbirine yakınsa (<=5mm) komşu
        n = len(bounds)
        self.zone_neighbors = [[] for _ in range(n)]
        NEIGHBOR_THRESHOLD_PX = 15  # piksel cinsinden yakınlık eşiği

        for i in range(n):
            bi = bounds[i]
            ri = (bi["x"], bi["y"], bi["x"] + bi["w"], bi["y"] + bi["h"])
            for j in range(i + 1, n):
                bj = bounds[j]
                rj = (bj["x"], bj["y"], bj["x"] + bj["w"], bj["y"] + bj["h"])
                if self._rects_adjacent(ri, rj, NEIGHBOR_THRESHOLD_PX):
                    self.zone_neighbors[i].append(j)
                    self.zone_neighbors[j].append(i)

    @staticmethod
    def _rects_adjacent(r1, r2, threshold):
        """İki dikdörtgenin kenarları threshold mesafe içinde mi kontrol et."""
        x1, y1, x1b, y1b = r1
        x2, y2, x2b, y2b = r2
        
        # Yatay örtüşme var mı?
        h_overlap = max(0, min(x1b, x2b) - max(x1, x2))
        # Dikey örtüşme var mı?
        v_overlap = max(0, min(y1b, y2b) - max(y1, y2))
        
        # Yatay komşuluk: dikey örtüşme var + yatay mesafe küçük
        h_gap = max(x1, x2) - min(x1b, x2b)
        if v_overlap > 0 and 0 <= h_gap <= threshold:
            return True
        
        # Dikey komşuluk: yatay örtüşme var + dikey mesafe küçük
        v_gap = max(y1, y2) - min(y1b, y2b)
        if h_overlap > 0 and 0 <= v_gap <= threshold:
            return True
        
        return False

    # ===== Ağırlık Hesaplama =====
    def calculate_total_weight(self, zone_results: Dict[int, Dict] = None) -> Dict[str, Any]:
        """
        Toplam panel ağırlığı hesapla.
        
        Weight = Σ (Area_i × PlyCount_i × PLY_THICKNESS_MM × CFRP_DENSITY_G_MM3)
        
        Returns:
            {"total_weight_g": float, "zone_weights_g": list, "has_geometry": bool}
        """
        if not self.zone_areas_mm2:
            # Geometri yoksa sadece ply bazlı oransal ağırlık döndür
            zone_weights = []
            for i, total in enumerate(self.zone_totals):
                # Alan bilinmiyorsa birim alan (1 mm²) varsay
                w = total * PLY_THICKNESS_MM * CFRP_DENSITY_G_MM3
                zone_weights.append(w)
            return {
                "total_weight_g": sum(zone_weights),
                "zone_weights_g": zone_weights,
                "has_geometry": False
            }
        
        zone_weights = []
        for i, area_mm2 in enumerate(self.zone_areas_mm2):
            ply_count = self.zone_totals[i]
            thickness_mm = ply_count * PLY_THICKNESS_MM
            volume_mm3 = area_mm2 * thickness_mm
            weight_g = volume_mm3 * CFRP_DENSITY_G_MM3
            zone_weights.append(round(weight_g, 3))
        
        return {
            "total_weight_g": round(sum(zone_weights), 3),
            "zone_weights_g": zone_weights,
            "has_geometry": True
        }

    # ===== Ramp Kısıtı Kontrolü =====
    def check_ramp_feasibility(self) -> List[Dict[str, Any]]:
        """
        Komşu zone'lar arasındaki ply farkı için yeterli fiziksel mesafe var mı kontrol et.
        
        Kural: Required_Ramp_Length = |PlyDiff| × RAMP_RATE_MM_PER_PLY
        Kısıt: Zone'un komşuya bakan kenarının uzunluğu >= Required_Ramp_Length
        
        Returns:
            List of ramp check results (violations included)
        """
        results = []
        
        if not self.zone_dims_mm or not self.zone_neighbors:
            return results  # Geometri yoksa kontrol yapılamaz
        
        checked_pairs = set()
        
        for i in range(len(self.zones_config)):
            for j in self.zone_neighbors[i]:
                pair = tuple(sorted([i, j]))
                if pair in checked_pairs:
                    continue
                checked_pairs.add(pair)
                
                ply_diff = abs(self.zone_totals[i] - self.zone_totals[j])
                required_ramp_mm = ply_diff * RAMP_RATE_MM_PER_PLY
                
                # Her iki zone'un da en kısa kenarı ramp mesafesini karşılamalı
                min_dim_i = min(self.zone_dims_mm[i])
                min_dim_j = min(self.zone_dims_mm[j])
                available_mm = min(min_dim_i, min_dim_j)
                
                feasible = available_mm >= required_ramp_mm
                
                results.append({
                    "zone_a": i,
                    "zone_b": j,
                    "ply_diff": ply_diff,
                    "required_ramp_mm": round(required_ramp_mm, 2),
                    "available_mm": round(available_mm, 2),
                    "feasible": feasible,
                    "margin_mm": round(available_mm - required_ramp_mm, 2)
                })
        
        return results

    def optimize_all(self, progress_callback=None) -> Dict[str, Any]:
        """
        Tüm zone'ları optimize et.
        
        Returns:
            {
                "zones": [
                    {"index": 0, "sequence": [...], "ply_count": 36, "fitness": 95.2, "is_root": True},
                    {"index": 1, "sequence": [...], "ply_count": 32, "fitness": 93.1, "is_root": False},
                    ...
                ],
                "transitions": [
                    {"from": 0, "to": 1, "dropped_indices": [...]},
                    ...
                ],
                "root_updated": False,
                "total_iterations": 1
            }
        """
        print("=" * 60)
        print("MULTI-ZONE OPTIMIZATION")
        print("=" * 60)
        print(f"Zone sayısı: {len(self.zones_config)}")
        print(f"Root zone: Zone {self.root_index + 1} ({self.zone_totals[self.root_index]} ply)")
        print(f"Sıralama: {[f'Zone {i+1} ({self.zone_totals[i]} ply)' for i in self.sorted_zone_indices]}")
        print()

        root_updated = False
        total_iterations = 0
        
        # Callback wrapper
        def report_progress(val, msg=""):
            if progress_callback:
                progress_callback({"progress": val, "message": msg})

        report_progress(5, "Optimizasyon baslatiliyor...")
        
        for attempt in range(self.MAX_ROOT_RETRIES):
            total_iterations += 1
            print(f"\n--- Deneme {attempt + 1}/{self.MAX_ROOT_RETRIES} ---")
            
            # 1. Root zone'u optimize et
            root_config = self.zones_config[self.root_index]
            print(f"\nRoot Zone (Zone {self.root_index + 1}) optimizasyonu başlıyor...")
            
            report_progress(15, f"Root Zone (Zone {self.root_index + 1}) optimize ediliyor...")
            root_optimizer = LaminateOptimizer(root_config)
            root_seq, root_score, root_details, _ = root_optimizer.run_hybrid_optimization()
            
            print(f"Root Zone skor: {root_score:.2f}/100")
            report_progress(25, f"Root Zone tamamlandi (Skor: {root_score:.1f})")

            # Zone sonuçlarını sakla
            zone_results = {}
            zone_results[self.root_index] = {
                "index": self.root_index,
                "sequence": root_seq,
                "ply_count": len(root_seq),
                "fitness": root_score,
                "details": root_details,
                "is_root": True,
                "ply_counts": dict(Counter(root_seq))
            }

            transitions = []
            drop_success = True

            # 2. Diğer zone'ları drop-off ile türet (kalından inceye sırayla)
            for i, zone_idx in enumerate(self.sorted_zone_indices):
                if zone_idx == self.root_index:
                    continue  # Root zaten optimize edildi

                target_config = self.zones_config[zone_idx]
                target_total = sum(target_config.values())
                
                # Kaynak zone'u bul (bir önceki kalın zone)
                source_idx = self.sorted_zone_indices[self.sorted_zone_indices.index(zone_idx) - 1]
                source_result = zone_results[source_idx]
                source_seq = source_result["sequence"]
                
                print(f"\nZone {zone_idx + 1} ({target_total} ply) - Zone {source_idx + 1}'den drop-off yapılıyor...")

                # Drop-off optimizer oluştur
                source_optimizer = LaminateOptimizer(dict(Counter(source_seq)))
                drop_optimizer = DropOffOptimizer(source_seq, source_optimizer)

                try:
                    # Açıya özel drop-off yap
                    new_seq, drop_score, dropped_by_angle = drop_optimizer.optimize_drop_with_angle_targets(target_config)
                    
                    # Fitness hesapla
                    target_optimizer = LaminateOptimizer(target_config)
                    fitness, details = target_optimizer.calculate_fitness(new_seq)

                    # Drop-off sonrası kısa local search ile kaliteyi artır
                    if fitness > 0:
                        polished_seq, polished_score = target_optimizer._local_search(new_seq, max_iter=25)
                        if polished_score > fitness:
                            new_seq = polished_seq
                            fitness = polished_score
                            _, details = target_optimizer.calculate_fitness(new_seq)
                            print(f"  Zone {zone_idx + 1} polish: {polished_score:.2f}/100")

                    zone_results[zone_idx] = {
                        "index": zone_idx,
                        "sequence": new_seq,
                        "ply_count": len(new_seq),
                        "fitness": float(fitness),
                        "details": details,
                        "is_root": False,
                        "ply_counts": dict(Counter(new_seq)),
                        "dropped_by_angle": dropped_by_angle
                    }
                    
                    # Transition kaydet
                    all_dropped = []
                    for angle, indices in dropped_by_angle.items():
                        all_dropped.extend(indices)
                    
                    transitions.append({
                        "from": source_idx,
                        "to": zone_idx,
                        "dropped_indices": sorted(all_dropped),
                        "dropped_by_angle": dropped_by_angle
                    })
                    
                    print(f"Zone {zone_idx + 1} skor: {fitness:.2f}/100")
                    
                    # Calculate progress for zones (25% to 90%)
                    total_zones = len(self.sorted_zone_indices) - 1 # excluding root
                    if total_zones > 0:
                        # i is 0-indexed in loop, but loop skips root, so we track count
                        # Actually enumerate gives 0, 1, 2...
                        # Let's use simple logic:
                        percent_per_zone = 65.0 / total_zones
                        current_progress = 25 + ((i + 1) * percent_per_zone)
                        report_progress(int(current_progress), f"Zone {zone_idx + 1} tamamlandi ({fitness:.1f})")

                except Exception as e:
                    print(f"Zone {zone_idx + 1} drop-off BAŞARISIZ: {e}")
                    drop_success = False
                    break

            if drop_success:
                # Tüm zone'lar başarılı
                print("\n" + "=" * 60)
                print("MULTI-ZONE OPTIMIZATION TAMAMLANDI")
                print("=" * 60)
                
                # Sonuçları sırala (zone index'e göre)
                sorted_results = [zone_results[i] for i in range(len(self.zones_config))]
                
                # Ağırlık hesaplama
                weight_info = self.calculate_total_weight(zone_results)
                if weight_info["has_geometry"]:
                    report_progress(95, "Agirlik ve kisit kontrolleri yapiliyor...")
                    print(f"Toplam ağırlık: {weight_info['total_weight_g']:.2f} g")
                    for idx, wg in enumerate(weight_info["zone_weights_g"]):
                        print(f"  Zone {idx+1}: {wg:.2f} g")
                
                # Ramp kısıtı kontrolü
                ramp_checks = self.check_ramp_feasibility()
                ramp_violations = [r for r in ramp_checks if not r["feasible"]]
                if ramp_violations:
                    print(f"\nUYARI: {len(ramp_violations)} ramp kısıtı ihlali!")
                    for v in ramp_violations:
                        print(f"  Zone {v['zone_a']+1} <-> Zone {v['zone_b']+1}: "
                              f"{v['ply_diff']} ply fark, "
                              f"gereken={v['required_ramp_mm']:.1f}mm, "
                              f"mevcut={v['available_mm']:.1f}mm")
                elif ramp_checks:
                    print("Tum ramp kisitlari karsilaniyor.")
                
                return {
                    "success": True,
                    "zones": sorted_results,
                    "transitions": transitions,
                    "root_updated": root_updated,
                    "total_iterations": total_iterations,
                    "root_index": self.root_index,
                    "weight": weight_info,
                    "ramp_checks": ramp_checks
                }
            else:
                # Drop-off başarısız - root'u güncelle
                print("\nDrop-off başarısız, root zone yeniden optimize ediliyor...")
                root_updated = True
                # Yeni deneme için devam et

        # Maksimum deneme aşıldı
        print("\nMAKSİMUM DENEME AŞILDI - Kısmi sonuç döndürülüyor")
        report_progress(100, "Islem tamamlandi (Maksimum deneme asildi)")
        sorted_results = [zone_results.get(i, None) for i in range(len(self.zones_config))]
        
        return {
            "success": False,
            "zones": sorted_results,
            "transitions": transitions,
            "root_updated": root_updated,
            "total_iterations": total_iterations,
            "root_index": self.root_index,
            "error": "Maksimum deneme sayısı aşıldı"
        }

    def get_zone_summary(self) -> str:
        """Zone konfigürasyonlarının özetini döndür."""
        lines = []
        for i, config in enumerate(self.zones_config):
            total = sum(config.values())
            is_root = i == self.root_index
            root_marker = " (ROOT)" if is_root else ""
            lines.append(f"Zone {i + 1}: {total} ply{root_marker} - 0°:{config.get(0,0)}, 90°:{config.get(90,0)}, +45°:{config.get(45,0)}, -45°:{config.get(-45,0)}")
        return "\n".join(lines)
