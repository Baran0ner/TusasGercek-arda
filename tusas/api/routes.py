import time
from collections import Counter

import os
import json
import threading
import queue
from flask import Blueprint, jsonify, request, send_from_directory, Response, stream_with_context

from ..core.dropoff_optimizer import DropOffOptimizer
from ..core.laminate_optimizer import LaminateOptimizer
from ..core.multi_zone_optimizer import MultiZoneOptimizer
from ..core.symmetry import check_symmetry_compatibility
from ..state import get_zone_manager, set_zone_manager
from ..zones.manager import ZoneManager
from ..zones.models import Zone


bp = Blueprint("tusas_api", __name__)


@bp.route("/")
def index():
    # Always serve from project root (avoid cwd/encoding issues on Windows paths)
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    return send_from_directory(root_dir, "index.html")


@bp.post("/optimize")
def optimize():
    payload = request.get_json(force=True, silent=True) or {}
    ply_counts = payload.get("ply_counts", {})
    ply_counts = {int(k): int(v) for k, v in ply_counts.items() if str(v).isdigit()} or {0: 18, 90: 18, 45: 18, -45: 18}

    population_size = int(payload.get("population_size", 120))
    generations = int(payload.get("generations", 600))
    min_drop = int(payload.get("min_drop", 48))
    drop_step = int(payload.get("drop_step", 8))

    symmetry_check = check_symmetry_compatibility(ply_counts)
    user_choice = payload.get("symmetry_user_choice")
    if symmetry_check["requires_user_choice"] and not user_choice:
        return jsonify(
            {
                "requires_symmetry_choice": True,
                "symmetry_info": symmetry_check,
                "message": "Simetri uyarısı: Kullanıcı seçimi gerekiyor",
            }
        ), 200

    if user_choice:
        if user_choice.get("continue_with_current"):
            pass
        elif "adjusted_counts" in user_choice:
            ply_counts = user_choice["adjusted_counts"]

    total_plies = sum(ply_counts.values())
    if population_size <= 120:
        population_size = max(120, min(400, int(total_plies * 2.0)))
    if generations <= 600:
        generations = max(600, min(1500, int(total_plies * 10.0)))

    optimizer = LaminateOptimizer(ply_counts)
    start_time = time.time()
    master_seq, master_score, details, history = optimizer.run_hybrid_optimization()
    ga_elapsed = time.time() - start_time

    drop_targets = []
    temp = len(master_seq)
    while temp > min_drop:
        temp -= drop_step
        if temp > 0:
            drop_targets.append(temp)

    drop_opt = DropOffOptimizer(master_seq, optimizer)
    drop_results_list = []
    current_seq = master_seq
    for target in drop_targets:
        drop_opt.master_sequence = current_seq
        drop_opt.total_plies = len(current_seq)
        new_seq, sc, dropped_indices = drop_opt.optimize_drop(target)
        drop_results_list.append({"target": target, "seq": new_seq, "score": sc, "dropped": dropped_indices})
        current_seq = new_seq

    response = {
        "master_sequence": master_seq,
        "fitness_score": details.get("total_score", master_score),
        "max_score": details.get("max_score", 100),
        "penalties": details.get("rules", {}),
        "history": history,
        "drop_off_results": drop_results_list,
        "stats": {
            "plies": len(master_seq),
            "duration_seconds": round(ga_elapsed, 2),
            "population_size": population_size,
            "generations": generations,
        },
    }
    return jsonify(response)


@bp.post("/optimize_multi_zone")
def optimize_multi_zone():
    """
    Çoklu zone optimizasyonu endpoint'i.
    
    Request body:
    {
        "zones": [
            {"0": 12, "90": 8, "45": 8, "-45": 8},
            {"0": 8, "90": 8, "45": 8, "-45": 8},
            {"0": 6, "90": 6, "45": 6, "-45": 6}
        ]
    }
    """
    payload = request.get_json(force=True, silent=True) or {}
    zones = payload.get("zones", [])

    if not zones or len(zones) < 2:
        return jsonify({"error": "En az 2 zone gerekli"}), 400

    # Zone formatını doğrula
    for i, zone in enumerate(zones):
        if not isinstance(zone, dict):
            return jsonify({"error": f"Zone {i + 1} geçersiz format"}), 400
        try:
            total = sum(int(v) for v in zone.values())
            if total <= 0:
                return jsonify({"error": f"Zone {i + 1} boş olamaz"}), 400
        except (ValueError, TypeError):
            return jsonify({"error": f"Zone {i + 1} geçersiz değerler içeriyor"}), 400

    # Opsiyonel: zone konumları (Panel Designer'dan)
    bounds = payload.get("bounds", None)  # [{"x":..,"y":..,"w":..,"h":..}, ...]
    panel_scale_mm = float(payload.get("panel_scale_mm", 300))
    # Opsiyonel: kural agirliklari (R1..R8 oranlari)
    rule_weights = payload.get("rule_weights", None)  # {"R1": 18, "R2": 12, ...}

    start_time = time.time()
    
    try:
        optimizer = MultiZoneOptimizer(zones, bounds=bounds, panel_scale_mm=panel_scale_mm, rule_weights=rule_weights)
        result = optimizer.optimize_all()
    except Exception as e:
        # Hata mesajını UTF-8 güvenli şekilde al (Windows charmap hatası önlenir)
        err_msg = str(e).encode("utf-8", errors="replace").decode("utf-8")
        return jsonify({"error": f"Optimizasyon hatası: {err_msg}"}), 500

    elapsed = time.time() - start_time

    # Bağlantısız zone hatası kontrolü
    if not result.get("success") and result.get("disconnected_zones"):
        return jsonify({
            "error": result["error"],
            "disconnected_zones": result["disconnected_zones"],
            "root_index": result.get("root_index", 0),
            "neighbor_graph": result.get("neighbor_graph", []),
        }), 400

    # Sonuçları frontend için formatla
    zone_results = []
    for zone_data in result.get("zones", []):
        if zone_data is None:
            zone_results.append(None)
            continue
        zone_results.append({
            "index": zone_data["index"],
            "sequence": zone_data["sequence"],
            "ply_count": zone_data["ply_count"],
            "fitness": zone_data["fitness"],
            "is_root": zone_data["is_root"],
            "ply_counts": zone_data.get("ply_counts", {}),
            "penalties": zone_data.get("details", {}).get("rules", {})
        })

    # Ağırlık ve ramp bilgileri
    weight_info = result.get("weight", {})
    ramp_checks = result.get("ramp_checks", [])

    return jsonify({
        "success": result.get("success", False),
        "zones": zone_results,
        "transitions": result.get("transitions", []),
        "root_index": result.get("root_index", 0),
        "weight": weight_info,
        "ramp_checks": ramp_checks,
        "drop_off_tree": result.get("drop_off_tree", {}),
        "neighbor_graph": result.get("neighbor_graph", []),
        "stats": {
            "duration_seconds": round(elapsed, 2),
            "total_iterations": result.get("total_iterations", 1),
            "root_updated": result.get("root_updated", False)
        }
    })


@bp.post("/optimize_multi_zone_stream")
def optimize_multi_zone_stream():
    """
    SSE ile gercek zamanli ilerleme bildiren optimizasyon endpoint'i.
    """
    payload = request.get_json(force=True, silent=True) or {}
    zones = payload.get("zones", [])

    if not zones or len(zones) < 2:
        return jsonify({"error": "En az 2 zone gerekli"}), 400

    # Zone formatını doğrula
    for i, zone in enumerate(zones):
        if not isinstance(zone, dict):
            return jsonify({"error": f"Zone {i + 1} geçersiz format"}), 400
        try:
            total = sum(int(v) for v in zone.values())
            if total <= 0:
                return jsonify({"error": f"Zone {i + 1} boş olamaz"}), 400
        except (ValueError, TypeError):
            return jsonify({"error": f"Zone {i + 1} geçersiz değerler içeriyor"}), 400

    bounds = payload.get("bounds", None)
    panel_scale_mm = float(payload.get("panel_scale_mm", 300))
    rule_weights = payload.get("rule_weights", None)

    # Thread-safe queue
    q = queue.Queue()

    def worker():
        try:
            optimizer = MultiZoneOptimizer(zones, bounds=bounds, panel_scale_mm=panel_scale_mm, rule_weights=rule_weights)
            
            def progress_callback(data):
                q.put({"type": "progress", "data": data})

            # Optimizasyonu calistir
            start_time = time.time()
            result = optimizer.optimize_all(progress_callback=progress_callback)
            elapsed = time.time() - start_time

            # Sonuclari hazirla (mevcut mantikla ayni)
            zone_results = []
            for zone_data in result.get("zones", []):
                if zone_data is None:
                    zone_results.append(None)
                    continue
                zone_results.append({
                    "index": zone_data["index"],
                    "sequence": zone_data["sequence"],
                    "ply_count": zone_data["ply_count"],
                    "fitness": zone_data["fitness"],
                    "is_root": zone_data["is_root"],
                    "ply_counts": zone_data.get("ply_counts", {}),
                    "penalties": zone_data.get("details", {}).get("rules", {})
                })

            final_data = {
                "success": result.get("success", False),
                "zones": zone_results,
                "transitions": result.get("transitions", []),
                "root_index": result.get("root_index", 0),
                "weight": result.get("weight", {}),
                "ramp_checks": result.get("ramp_checks", []),
                "drop_off_tree": result.get("drop_off_tree", {}),
                "neighbor_graph": result.get("neighbor_graph", []),
                "disconnected_zones": result.get("disconnected_zones", []),
                "stats": {
                    "duration_seconds": round(elapsed, 2),
                    "total_iterations": result.get("total_iterations", 1),
                    "root_updated": result.get("root_updated", False)
                }
            }
            q.put({"type": "result", "data": final_data})

        except Exception as e:
            err_msg = str(e)
            q.put({"type": "error", "message": err_msg})
        finally:
            q.put(None) # Sentinel

    # Thread baslat
    t = threading.Thread(target=worker)
    t.start()

    def generate():
        while True:
            item = q.get()
            if item is None:
                break
            
            # SSE format: data: <json_string>\n\n
            yield f"data: {json.dumps(item)}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')


@bp.post("/evaluate")
def evaluate():
    payload = request.get_json(force=True, silent=True) or {}
    sequence = payload.get("sequence", [])
    ply_counts = payload.get("ply_counts", {})

    if not sequence:
        return jsonify({"error": "Sequence required"}), 400

    if not ply_counts:
        ply_counts = dict(Counter(sequence))

    try:
        sequence = [int(x) for x in sequence]
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid sequence format"}), 400

    optimizer = LaminateOptimizer(ply_counts)
    fitness_score, details = optimizer.calculate_fitness(sequence)
    fitness_score = float(fitness_score)

    return jsonify(
        {
            "sequence": sequence,
            "fitness_score": fitness_score,
            "max_score": details.get("max_score", 100),
            "penalties": details.get("rules", {}),
            "valid": bool(fitness_score > 0),
        }
    )


@bp.post("/dropoff")
def dropoff():
    payload = request.get_json(force=True, silent=True) or {}
    master_sequence = payload.get("master_sequence", [])
    target_ply = payload.get("target_ply")
    ply_counts = payload.get("ply_counts", {})

    if not master_sequence:
        return jsonify({"error": "master_sequence required"}), 400
    if target_ply is None:
        return jsonify({"error": "target_ply required"}), 400

    try:
        master_sequence = [int(x) for x in master_sequence]
        target_ply = int(target_ply)
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid format"}), 400

    if not ply_counts:
        ply_counts = dict(Counter(master_sequence))
    else:
        ply_counts = {int(k): int(v) for k, v in ply_counts.items() if str(v).isdigit()}

    current_ply = len(master_sequence)
    if target_ply >= current_ply:
        return jsonify({"error": "target_ply ({}) must be less than current ply count ({})".format(target_ply, current_ply)}), 400
    if target_ply <= 0:
        return jsonify({"error": "target_ply must be greater than 0"}), 400

    optimizer = LaminateOptimizer(ply_counts)
    drop_opt = DropOffOptimizer(master_sequence, optimizer)
    new_seq, score, dropped_indices = drop_opt.optimize_drop(target_ply)

    fitness_score, details = optimizer.calculate_fitness(new_seq)

    return jsonify(
        {
            "sequence": new_seq,
            "fitness_score": fitness_score,
            "max_score": details.get("max_score", 100),
            "penalties": details.get("rules", {}),
            "dropped_indices": dropped_indices,
            "target_ply": target_ply,
            "original_ply": current_ply,
            "removed_count": len(dropped_indices),
        }
    )


@bp.post("/dropoff_angle_targets")
def dropoff_angle_targets():
    payload = request.get_json(force=True, silent=True) or {}
    master_sequence = payload.get("master_sequence", [])
    target_ply_counts = payload.get("target_ply_counts", {})
    ply_counts = payload.get("ply_counts", {})

    if not master_sequence:
        return jsonify({"error": "master_sequence required"}), 400
    if not target_ply_counts:
        return jsonify({"error": "target_ply_counts required"}), 400

    try:
        master_sequence = [int(x) for x in master_sequence]
        target_ply_counts = {int(k): int(v) for k, v in target_ply_counts.items()}
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid format"}), 400

    if not ply_counts:
        ply_counts = dict(Counter(master_sequence))
    else:
        ply_counts = {int(k): int(v) for k, v in ply_counts.items() if str(v).isdigit()}

    current_ply_counts = dict(Counter(master_sequence))

    for angle, target_count in target_ply_counts.items():
        current = current_ply_counts.get(angle, 0)
        if target_count > current:
            return jsonify({"error": "Angle {}°: hedef {} ama mevcut sadece {} katman var".format(angle, target_count, current)}), 400

    optimizer = LaminateOptimizer(ply_counts)
    drop_opt = DropOffOptimizer(master_sequence, optimizer)

    try:
        new_seq, score, dropped_by_angle = drop_opt.optimize_drop_with_angle_targets(target_ply_counts)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    fitness_score, details = optimizer.calculate_fitness(new_seq)
    new_ply_counts = dict(Counter(new_seq))
    total_removed = len(master_sequence) - len(new_seq)

    return jsonify(
        {
            "sequence": new_seq,
            "fitness_score": fitness_score,
            "max_score": details.get("max_score", 100),
            "penalties": details.get("rules", {}),
            "dropped_by_angle": dropped_by_angle,
            "target_ply_counts": target_ply_counts,
            "current_ply_counts": current_ply_counts,
            "new_ply_counts": new_ply_counts,
            "total_removed": total_removed,
            "original_total": len(master_sequence),
            "new_total": len(new_seq),
        }
    )


@bp.post("/auto_optimize")
def auto_optimize():
    payload = request.get_json(force=True, silent=True) or {}
    ply_counts = payload.get("ply_counts", {})
    ply_counts = {int(k): int(v) for k, v in ply_counts.items() if str(v).isdigit()} or {0: 18, 90: 18, 45: 18, -45: 18}

    runs = int(payload.get("runs", 10))
    population_size = int(payload.get("population_size", 180))
    generations = int(payload.get("generations", 800))
    stagnation_window = int(payload.get("stagnation_window", 150))

    total_plies = sum(ply_counts.values())
    if population_size <= 180:
        population_size = max(180, min(400, int(total_plies * 2.0)))
    if generations <= 800:
        generations = max(800, min(1500, int(total_plies * 10.0)))

    optimizer = LaminateOptimizer(ply_counts)
    start_time = time.time()
    result = optimizer.auto_optimize(
        runs=runs, population_size=population_size, generations=generations, stagnation_window=stagnation_window
    )
    elapsed = time.time() - start_time

    optimizer2 = LaminateOptimizer(ply_counts)
    _, fitness_details = optimizer2.calculate_fitness(result["best_sequence"])

    return jsonify(
        {
            "master_sequence": result["best_sequence"],
            "fitness_score": fitness_details.get("total_score", result["best_fitness"]),
            "max_score": fitness_details.get("max_score", 100),
            "penalties": fitness_details.get("rules", {}),
            "history": result["history"],
            "stats": {
                "plies": len(result["best_sequence"]),
                "duration_seconds": round(elapsed, 2),
                "runs": runs,
                "population_size": population_size,
                "generations": generations,
                "stagnation_window": stagnation_window,
            },
        }
    )


# -----------------------------------------------------------------------------
# Zone Management Endpoints
# -----------------------------------------------------------------------------


@bp.post("/zones/init_root")
def init_root_zone():
    payload = request.get_json(force=True, silent=True) or {}
    session_id = payload.get("session_id", "default")
    master_sequence = payload.get("master_sequence", [])
    ply_counts = payload.get("ply_counts", {})

    if not master_sequence:
        return jsonify({"error": "master_sequence required"}), 400

    try:
        master_sequence = [int(x) for x in master_sequence]
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid sequence format"}), 400

    if not ply_counts:
        ply_counts = dict(Counter(master_sequence))

    zone_manager = ZoneManager()
    optimizer = LaminateOptimizer(ply_counts)
    zone_manager.add_root_zone(master_sequence, optimizer)
    set_zone_manager(session_id, zone_manager)

    root_zone = zone_manager.get_zone(0)
    return jsonify(
        {
            "success": True,
            "zone": root_zone.to_dict() if root_zone else None,
            "all_zones": zone_manager.get_all_zones(),
            "transitions": zone_manager.get_transitions(),
        }
    )


@bp.get("/zones/list")
def list_zones():
    session_id = request.args.get("session_id", "default")
    zm = get_zone_manager(session_id)
    if not zm:
        return jsonify({"zones": [], "transitions": [], "message": "No zones found. Initialize root zone first."})

    return jsonify({"zones": zm.get_all_zones(), "transitions": zm.get_transitions()})


@bp.get("/zones/<int:zone_id>")
def get_zone(zone_id: int):
    session_id = request.args.get("session_id", "default")
    zm = get_zone_manager(session_id)
    if not zm:
        return jsonify({"error": "Session not found"}), 404
    zone = zm.get_zone(zone_id)
    if not zone:
        return jsonify({"error": "Zone {} not found".format(zone_id)}), 404
    return jsonify({"zone": zone.to_dict(), "transitions": zm.get_transitions()})


@bp.post("/zones/create_from_dropoff")
def create_zone_from_dropoff():
    payload = request.get_json(force=True, silent=True) or {}
    session_id = payload.get("session_id", "default")
    source_zone_id = payload.get("source_zone_id")
    target_ply = payload.get("target_ply")
    ply_counts = payload.get("ply_counts", {})

    if source_zone_id is None:
        return jsonify({"error": "source_zone_id required"}), 400
    if target_ply is None:
        return jsonify({"error": "target_ply required"}), 400

    zm = get_zone_manager(session_id)
    if not zm:
        return jsonify({"error": "Session not found. Create root zone first."}), 400

    source_zone = zm.get_zone(source_zone_id)
    if not source_zone:
        return jsonify({"error": "Source zone {} not found".format(source_zone_id)}), 404

    if not ply_counts:
        ply_counts = dict(Counter(source_zone.sequence))

    optimizer = LaminateOptimizer(ply_counts)
    drop_optimizer = DropOffOptimizer(source_zone.sequence, optimizer)

    try:
        new_zone = zm.create_zone_from_dropoff(source_zone_id, int(target_ply), optimizer, drop_optimizer)
        return jsonify({"success": True, "zone": new_zone.to_dict(), "transitions": zm.get_transitions()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@bp.post("/zones/create_from_angle_dropoff")
def create_zone_from_angle_dropoff():
    payload = request.get_json(force=True, silent=True) or {}
    session_id = payload.get("session_id", "default")
    source_zone_id = payload.get("source_zone_id")
    target_ply_counts = payload.get("target_ply_counts", {})
    ply_counts = payload.get("ply_counts", {})

    if source_zone_id is None:
        return jsonify({"error": "source_zone_id required"}), 400
    if not target_ply_counts:
        return jsonify({"error": "target_ply_counts required"}), 400

    zm = get_zone_manager(session_id)
    if not zm:
        return jsonify({"error": "Session not found. Create root zone first."}), 400

    source_zone = zm.get_zone(source_zone_id)
    if not source_zone:
        return jsonify({"error": "Source zone {} not found".format(source_zone_id)}), 404

    if not ply_counts:
        ply_counts = dict(Counter(source_zone.sequence))

    try:
        target_ply_counts = {int(k): int(v) for k, v in target_ply_counts.items()}
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid target_ply_counts format"}), 400

    optimizer = LaminateOptimizer(ply_counts)
    drop_optimizer = DropOffOptimizer(source_zone.sequence, optimizer)

    try:
        new_zone = zm.create_zone_from_angle_dropoff(source_zone_id, target_ply_counts, optimizer, drop_optimizer)
        return jsonify(
            {"success": True, "zone": new_zone.to_dict(), "zones": zm.get_all_zones(), "transitions": zm.get_transitions()}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@bp.post("/zones/create_from_merge")
def create_zone_from_merge():
    payload = request.get_json(force=True, silent=True) or {}
    session_id = payload.get("session_id", "default")
    source_zone_ids = payload.get("source_zone_ids", [])
    target_ply = payload.get("target_ply")
    ply_counts = payload.get("ply_counts", {})

    if not source_zone_ids:
        return jsonify({"error": "source_zone_ids required"}), 400

    zm = get_zone_manager(session_id)
    if not zm:
        return jsonify({"error": "Session not found. Create root zone first."}), 400

    for zid in source_zone_ids:
        if not zm.get_zone(zid):
            return jsonify({"error": "Source zone {} not found".format(zid)}), 404

    if not ply_counts and source_zone_ids:
        first_zone = zm.get_zone(source_zone_ids[0])
        if first_zone:
            ply_counts = dict(Counter(first_zone.sequence))

    optimizer = LaminateOptimizer(ply_counts)

    try:
        if target_ply is None:
            max_ply = max([zm.get_zone(zid).ply_count for zid in source_zone_ids])
            target_ply = max_ply
        new_zone = zm.create_zone_from_merge(source_zone_ids, int(target_ply), optimizer)
        return jsonify({"success": True, "zone": new_zone.to_dict(), "zones": zm.get_all_zones(), "transitions": zm.get_transitions()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@bp.post("/zones/add_from_dropoff")
def add_zone_from_dropoff():
    payload = request.get_json(force=True, silent=True) or {}
    session_id = payload.get("session_id", "default")
    source_zone_id = payload.get("source_zone_id")
    sequence = payload.get("sequence", [])
    ply_count = payload.get("ply_count")
    fitness_score = payload.get("fitness_score", 0.0)
    dropped_indices = payload.get("dropped_indices", [])

    if not sequence:
        return jsonify({"error": "sequence required"}), 400
    if ply_count is None:
        ply_count = len(sequence)
    if source_zone_id is None:
        return jsonify({"error": "source_zone_id required"}), 400

    zm = get_zone_manager(session_id)
    if not zm:
        return jsonify({"error": "Session not found. Create root zone first."}), 400

    source_zone = zm.get_zone(source_zone_id)
    if not source_zone:
        return jsonify({"error": "Source zone {} not found".format(source_zone_id)}), 404

    try:
        sequence = [int(x) for x in sequence]
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid sequence format"}), 400

    zone_id = zm.next_zone_id
    zm.next_zone_id += 1

    new_zone = Zone(zone_id=zone_id, name="Zone {}".format(zone_id), sequence=sequence, ply_count=int(ply_count))
    new_zone.fitness_score = fitness_score
    new_zone.source_zones = [source_zone_id]
    new_zone.transition_type = "drop_off"

    zm.zones[zone_id] = new_zone
    zm.transitions.append(
        {
            "from": source_zone_id,
            "to": zone_id,
            "type": "drop_off",
            "target_ply": int(ply_count),
            "dropped_indices": dropped_indices,
        }
    )

    return jsonify({"success": True, "zone": new_zone.to_dict(), "zones": zm.get_all_zones(), "transitions": zm.get_transitions()})

