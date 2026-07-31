from abaqus import *
from abaqusConstants import *
import __main__
import os
import csv
import section
import regionToolset
import part
import material
import assembly
import step
import interaction
import load
import mesh
import job
import visualization
import xyPlot
import displayGroupOdbToolset as dgo

# ---------- Yeoh 3rd-order coefficients (requested) ----------
C10 = 2.58
C20 = -0.20
C30 = 1.93E-02
# ------------------------------------------------------------

PROFILE_A = 1.0
PROFILE_B = 1.0
ENCLOSED_AREA_SCALE = 0.9
U2_VALUE = 50.0
NUM_CPUS = 12
NUM_GPUS = 0
MEMORY_PCT = 90
THRESHOLD_STRAIN = 0.5

# 在 mm-N-s 体系下可写成 1.10e-9 ~ 1.25e-9 tonne/mm³
MATERIAL_DENSITY = 1.15e-9       
# -----------------------------------------------------------------------

# ========================= 单位换算（mm-N-s）=========================
ENERGY_TO_KJ_M3 = 1.0e3   
SEA_TO_J_KG = 1.0e-6      
# ===========================================================================


BASE_DIR = os.getcwd()
GRAPH_DIR = os.path.join(BASE_DIR, 'graph')

VARIABLES = ['RF2', 'U2']


# ========================= 辅助函数 =========================

def get_model_dimensions(model_name):
    try:
        mdl = mdb.models[model_name]
        xs, ys, zs = [], [], []
        for inst in mdl.rootAssembly.instances.values():
            try:
                for n in inst.nodes:
                    x, y, z = n.coordinates
                    xs.append(float(x)); ys.append(float(y)); zs.append(float(z))
            except Exception as e:
                print("Warning in get_model_dimensions node loop: %s" % e)
                continue
        if not xs:
            return None
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        minz, maxz = min(zs), max(zs)
        sx = maxx - minx
        sy = maxy - miny
        sz = maxz - minz
        center = ((minx + maxx) / 2.0, (miny + maxy) / 2.0, (minz + maxz) / 2.0)
        return {'min': (minx, miny, minz), 'max': (maxx, maxy, maxz),
                'size': (sx, sy, sz), 'center': center}
    except Exception as e:
        print("Error in get_model_dimensions: %s" % e)
        return None


def _get_profile_a_from_model(model_name):
    try:
        mdl = mdb.models[model_name]
        profiles = getattr(mdl, 'profiles', {})
        for pobj in profiles.values():
            if hasattr(pobj, 'a'):
                try:
                    return float(getattr(pobj, 'a'))
                except Exception:
                    continue
    except Exception:
        pass
    return None


def get_beam_width(model_name):
    try:
        if PROFILE_B is not None:
            return float(PROFILE_B)
    except Exception:
        pass
    try:
        mdl = mdb.models[model_name]
        profiles = getattr(mdl, 'profiles', {})
        for pobj in profiles.values():
            if hasattr(pobj, 'b'):
                try:
                    return float(getattr(pobj, 'b'))
                except Exception:
                    continue
    except Exception:
        pass
    return None


def get_total_length(model_name):
    try:
        mdl = mdb.models[model_name]
        p = mdl.parts['PART-1']
        total_len = 0.0
        for elem in p.elements:
            try:
                nodes = elem.getNodes()
                if len(nodes) >= 2:
                    c1 = nodes[0].coordinates; c2 = nodes[1].coordinates
                    dx = float(c2[0]) - float(c1[0])
                    dy = float(c2[1]) - float(c1[1])
                    dz = float(c2[2]) - float(c1[2])
                    total_len += (dx*dx + dy*dy + dz*dz) ** 0.5
            except Exception:
                continue
        return total_len
    except Exception:
        return None


def orientation(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0
    return 1 if val > 0 else 2


def convex_hull(points):
    n = len(points)
    if n < 3:
        return []
    ymin = min(range(n), key=lambda i: (points[i][1], points[i][0]))
    hull = []
    p = ymin
    while True:
        hull.append(points[p])
        q = (p + 1) % n
        for i in range(n):
            if orientation(points[p], points[i], points[q]) == 2:
                q = i
        p = q
        if p == ymin:
            break
    return hull


def polygon_area(points):
    n = len(points)
    if n < 3:
        return 0.0
    a = 0.0
    for i in range(n):
        j = (i + 1) % n
        a += points[i][0] * points[j][1] - points[j][0] * points[i][1]
    return abs(a) / 2.0


def get_total_enclosed_area(model_name):
    try:
        mdl = mdb.models[model_name]
        pts_set = set()
        for inst in mdl.rootAssembly.instances.values():
            for n in inst.nodes:
                x, y, z = n.coordinates
                pts_set.add((float(x), float(y)))
        pts = list(pts_set)
        if len(pts) < 3:
            return 0.0
        hull = convex_hull(pts)
        if len(hull) < 3:
            return 0.0
        area = polygon_area(hull)
        scaled = float(area) * float(ENCLOSED_AREA_SCALE)
        print("get_total_enclosed_area: convex_area = %.6g, scale = %.3g, "
              "scaled_area = %.6g" % (area, ENCLOSED_AREA_SCALE, scaled))
        return scaled
    except Exception:
        return None


# ========================= 模型构建 =========================

def import_model(inp_path, model_name):
    mdb.ModelFromInputFile(name=model_name, inputFileName=inp_path)
    if 'Model-1' in mdb.models:
        try:
            del mdb.models['Model-1']
        except Exception:
            pass


def define_material_and_section(model_name):
    try:
        mdl = mdb.models[model_name]
        p = mdl.parts['PART-1']
    except Exception as e:
        print("ERROR in define_material_and_section: cannot get model/part: %s" % e)
        return

    try:
        if 'TPU-95A' in mdl.materials:
            del mdl.materials['TPU-95A']
        mdl.Material(name='TPU-95A')
        # ------------------ Yeoh 三阶本构 ------------------
        mdl.materials['TPU-95A'].Hyperelastic(
            materialType=ISOTROPIC, testData=OFF, type=YEOH,
            table=((C10, C20, C30),),
            volumetricResponse=VOLUMETRIC_DATA
        )
        # -----------------------------------------------------------
    except Exception as e:
        print("Error defining material: %s" % e)

    try:
        if 'RECTANGLE' in mdl.profiles:
            del mdl.profiles['RECTANGLE']
        mdl.RectangularProfile(name='RECTANGLE', a=PROFILE_A, b=PROFILE_B)
    except Exception as e:
        print("Error defining profile: %s" % e)

    try:
        if 'RECTANGLE' in mdl.sections:
            del mdl.sections['RECTANGLE']
        mdl.BeamSection(
            name='RECTANGLE',
            integration=DURING_ANALYSIS,
            poissonRatio=0.0,
            profile='RECTANGLE',
            material='TPU-95A',
            temperatureVar=LINEAR,
            consistentMassMatrix=False
        )
    except Exception as e:
        print("Error defining beam section: %s" % e)

    try:
        elems = p.elements
        region = regionToolset.Region(elements=elems)
        if hasattr(p, 'sectionAssignments') and p.sectionAssignments:
            for i in range(len(p.sectionAssignments) - 1, -1, -1):
                try:
                    del p.sectionAssignments[i]
                except Exception as e:
                    print("Warning: failed to delete sectionAssignment[%d]: %s" % (i, e))
        p.SectionAssignment(region=region, sectionName='RECTANGLE')
        p.assignBeamSectionOrientation(
            region=region, method=N1_COSINES, n1=(0.0, 0.0, -1.0)
        )
    except Exception as e:
        print("Error in section assignment: %s" % e)


# ========================= 装配与边界条件 =========================

def assemble_and_step(model_name):
    try:
        mdl = mdb.models[model_name]
        a = mdl.rootAssembly
        a.regenerate()
    except Exception as e:
        print("ERROR in assemble_and_step: cannot get assembly: %s" % e)
        return

    step_name = 'Step-1'

    for bc_name in list(mdl.boundaryConditions.keys()):
        try:
            del mdl.boundaryConditions[bc_name]
            print("Deleted existing BC: %s" % bc_name)
        except Exception as e:
            print("Warning: failed to delete BC '%s': %s" % (bc_name, e))

    for sname in list(mdl.steps.keys()):
        if sname == 'Initial':
            continue
        try:
            del mdl.steps[sname]
            print("Deleted existing step: %s" % sname)
        except Exception as e:
            print("Warning: failed to delete step '%s': %s" % (sname, e))

    try:
        mdl.StaticStep(
            name=step_name, previous='Initial', maxNumInc=10000,
            initialInc=0.001, minInc=1e-6, maxInc=0.05, nlgeom=ON
        )
    except Exception as e:
        print("ERROR creating step: %s" % e)
        return

    # -----------------创建 Smooth Step 幅值 -----------------
    amp_name = 'SMOOTH_STEP_AMP'
    try:
        if amp_name in mdl.amplitudes:
            try:
                del mdl.amplitudes[amp_name]
                print("Deleted existing amplitude: %s" % amp_name)
            except Exception as e:
                print("Warning: failed to delete existing amplitude %s: %s" % (amp_name, e))

        mdl.SmoothStepAmplitude(name=amp_name, timeSpan=STEP, data=((0.0, 0.0), (1.0, 1.0)))
        # mdl.SmoothStepAmplitude(name=amp_name, timeSpan=STEP, data=((0.0, 0.0), (0.3, 0.1), (0.5, 0.2), (0.6, 0.3), (1.0, 1.0)))
        print("Created SmoothStepAmplitude: %s with data ((0.0,0.0),(1.0,1.0))" % amp_name)
    except Exception as e:
        print("Warning: failed to create SmoothStepAmplitude '%s': %s" % (amp_name, e))
    # -------------------------------------------------------------

    if not a.instances:
        print("ERROR: No instances found in assembly for model %s" % model_name)
        return

    inst_name_1 = list(a.instances.keys())[0]
    inst1 = a.instances[inst_name_1]
    print("Using instance: %s" % inst_name_1)

    inst_keys = list(a.instances.keys())
    if len(inst_keys) > 1:
        print("WARNING: Found %d instances, keeping only '%s', deleting others."
              % (len(inst_keys), inst_name_1))
        for iname in inst_keys:
            if iname != inst_name_1:
                try:
                    del a.instances[iname]
                    print("  Deleted duplicate instance: %s" % iname)
                except Exception as e:
                    print("  Warning: failed to delete instance '%s': %s" % (iname, e))

    try:
        y_coords = [n.coordinates[1] for n in inst1.nodes]
        if not y_coords:
            print("ERROR: No nodes found in instance %s" % inst_name_1)
            return
        min_y = min(y_coords)
        max_y = max(y_coords)
        tol = 1e-6

        bottom = inst1.nodes.getByBoundingBox(
            xMin=-1e9, xMax=1e9,
            yMin=min_y - tol, yMax=min_y + tol,
            zMin=-1e9, zMax=1e9
        )
        if not bottom:
            print("WARNING: No bottom nodes found for model %s" % model_name)
        else:
            if 'BOTTOM' in a.sets:
                del a.sets['BOTTOM']
            bset = a.Set(nodes=bottom, name='BOTTOM')
            mdl.EncastreBC(name='FIX', createStepName='Initial', region=bset)
            print("Created EncastreBC on BOTTOM (%d nodes) at Initial step" % len(bottom))

        top = inst1.nodes.getByBoundingBox(
            xMin=-1e9, xMax=1e9,
            yMin=max_y - tol, yMax=max_y + tol,
            zMin=-1e9, zMax=1e9
        )
        if not top:
            print("WARNING: No top nodes found for model %s" % model_name)
        else:
            if 'TOP' in a.sets:
                del a.sets['TOP']
            tset = a.Set(nodes=top, name='TOP')
            mdl.DisplacementBC(
                name='BC-2', createStepName=step_name, region=tset,
                u1=0, u2=U2_VALUE, u3=0, ur1=UNSET, ur2=UNSET, ur3=UNSET,
                amplitude=amp_name
            )
            #                amplitude=amp_name
            print("Created DisplacementBC on TOP (%d nodes) at %s with amplitude %s"
                  % (len(top), step_name, amp_name))

    except Exception as e:
        print("ERROR in assemble_and_step BC creation: %s" % e)


# ========================= 提交任务 =========================

def submit_job(model_name, job_name):
    try:
        if job_name in mdb.jobs:
            try:
                del mdb.jobs[job_name]
            except Exception:
                pass
        j = mdb.Job(
            name=job_name, model=model_name, type=ANALYSIS,
            memory=MEMORY_PCT, memoryUnits=PERCENTAGE,
            explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE,
            multiprocessingMode=DEFAULT,
            numCpus=int(NUM_CPUS), numDomains=int(NUM_CPUS), numGPUs=int(NUM_GPUS)
        )
        j.submit(consistencyChecking=OFF)
        print("Job %s submitted, waiting for completion..." % job_name)
        j.waitForCompletion()
        print("Job %s completed successfully" % job_name)
        return True
    except Exception as e:
        print("Job %s failed: %s" % (job_name, e))
        return False


# ========================= 读取 XY 数据 =========================

def _get_xydata_points_from_session(name):
    try:
        if name not in session.xyDataObjects:
            return None
        xyobj = session.xyDataObjects[name]
        pts = None
        if hasattr(xyobj, 'data'):
            pts = xyobj.data
        elif hasattr(xyobj, 'values'):
            pts = xyobj.values
        else:
            try:
                pts = tuple(xyobj)
            except Exception:
                pts = None
        if not pts:
            return None

        normalized = []
        if hasattr(pts, '__iter__') and len(pts) > 0 and hasattr(pts[0], '__iter__'):
            for row in pts:
                try:
                    x = float(row[0]); y = float(row[1])
                    normalized.append((x, y))
                except Exception:
                    continue
        else:
            for idx, val in enumerate(pts):
                try:
                    normalized.append((float(idx), float(val)))
                except Exception:
                    continue
        return normalized
    except Exception as e:
        print("Error reading session.xyDataObjects[%s] : %s" % (name, e))
        return None


# ========================= CSV 输出 =========================

def save_stress_strain_csv(model_name, ss_pairs, energy_density_list,
                           sea_list, data_dir):                        
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    csv_out = os.path.join(data_dir, "%s.csv" % model_name)
    try:
        with open(csv_out, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['strain', 'stress(MPa)', 'energy_density(kJ/m^3)', 'SEA(J/kg)'])  
            n = min(len(ss_pairs), len(energy_density_list), len(sea_list))
            for i in range(n):
                try:
                    strain = abs(float(ss_pairs[i][0]))
                    stress_mpa = abs(float(ss_pairs[i][1]))
                    ed_kj_m3 = float(energy_density_list[i]) * ENERGY_TO_KJ_M3
                    sea_j_kg = float(sea_list[i]) * SEA_TO_J_KG
                    w.writerow([strain, stress_mpa, ed_kj_m3, sea_j_kg])
                except Exception:
                    continue
        print("Saved CSV: %s" % csv_out)
        return csv_out
    except Exception as e:
        print("Failed to save CSV for %s : %s" % (model_name, e))
        return None


# ========================= 核心：应力-应变 + 能量 + SEA =========================

def convert_xydata_to_stress_strain_and_save(model_name, data_dir,
                                             thickness_override=None):
    xy_RF = _get_xydata_points_from_session('RF2')
    xy_U  = _get_xydata_points_from_session('U2')
    if xy_RF is None or xy_U is None:
        print("RF2 or U2 not found in session.xyDataObjects for model: %s"
              % model_name)
        return None

    dict_RF = {round(x, 12): y for (x, y) in xy_RF}
    dict_U  = {round(x, 12): y for (x, y) in xy_U}
    common_x = sorted(set(dict_RF.keys()) & set(dict_U.keys()))
    use_index_pairing = False
    if not common_x:
        if len(xy_RF) == len(xy_U):
            use_index_pairing = True
            common_x = [round(xy_RF[i][0], 12) for i in range(len(xy_RF))]
        else:
            common_x = sorted(dict_RF.keys())

    dims = get_model_dimensions(model_name)
    if dims is None:
        print("Cannot get model dimensions for %s" % model_name)
        return None

    sx, sy, sz = dims['size']

    if thickness_override is not None:
        try:
            thickness = float(thickness_override)
        except Exception:
            thickness = None
    else:
        prof_a = _get_profile_a_from_model(model_name)
        thickness = float(prof_a) if prof_a is not None else (
            float(sz) if sz is not None else None)

    if thickness is None or thickness == 0.0 or sx is None or sx == 0.0:
        print("Invalid thickness or sx for model %s  sx: %s  thickness: %s"
              % (model_name, sx, thickness))
        return None

    A  = float(sx) * float(thickness)   
    L0 = float(sy)                     
    if L0 == 0.0:
        print("Invalid L0 for %s" % model_name)
        return None

    V_envelope = float(sx) * float(sy) * float(thickness)   

    L_total = get_total_length(model_name)                 
    prof_a_val = float(PROFILE_A)
    prof_b_val = float(PROFILE_B)
    if L_total and L_total > 0:
        V_material = L_total * prof_a_val * prof_b_val       
        mass = float(MATERIAL_DENSITY) * V_material           
    else:
        V_material = None
        mass = None

    if mass and mass > 0:
        sea_factor = V_envelope / mass
        print("SEA factor = V_envelope / mass = %.6g / %.6g = %.6g"
              % (V_envelope, mass, sea_factor))
    else:
        sea_factor = None
        print("WARNING: Cannot compute SEA — mass is zero or unavailable "
              "(L_total=%s, density=%s)" % (L_total, MATERIAL_DENSITY))
    # ----------------------------------------------------------------

    ss_signed = []
    for i, x in enumerate(common_x):
        if use_index_pairing:
            fval = float(xy_RF[i][1]) if len(xy_RF) > i else 0.0
            uval = float(xy_U[i][1]) if len(xy_U) > i else 0.0
        else:
            fval = dict_RF.get(x, 0.0)
            uval = dict_U.get(x, 0.0)

        try:
            strain_signed = float(uval) / L0
        except Exception:
            strain_signed = 0.0
        try:
            stress_abs = abs(float(fval)) / A if A else 0.0
        except Exception:
            stress_abs = 0.0
        ss_signed.append((strain_signed, stress_abs))

    if not ss_signed:
        print("No valid signed stress-strain points for %s" % model_name)
        return None

    uniq = {}
    for s, st in ss_signed:
        k = round(float(s), 12)
        uniq[k] = (float(s), float(st))
    sorted_signed = sorted(uniq.values(), key=lambda t: t[0])

    Threshold = float(THRESHOLD_STRAIN)
    if Threshold < 0.0:
        print("ERROR: Threshold (THRESHOLD_STRAIN) is negative (%.6g). "
              "Aborting for model %s." % (Threshold, model_name))
        return None
    do_threshold_ops = (Threshold > 0.0)

    max_pos = max((s for s, _ in sorted_signed), default=0.0)
    min_neg = min((s for s, _ in sorted_signed), default=0.0)
    if abs(min_neg) > abs(max_pos):
        direction = 'compression'
        extreme_mag = abs(min_neg)
    else:
        direction = 'tension'
        extreme_mag = max_pos

    if not do_threshold_ops:
        abs_pairs = [(abs(s), st) for (s, st) in sorted_signed]
        abs_pairs_sorted = sorted(abs_pairs, key=lambda t: t[0])
        truncated_pairs = abs_pairs_sorted
    else:
        if direction == 'tension':
            if max_pos < Threshold - 1e-15:
                print("Max positive strain (%.6g) < Threshold (%.6g): "
                      "skipping CSV generation for model %s"
                      % (max_pos, Threshold, model_name))
                return None
            pos_list = [(s, st) for (s, st) in sorted_signed if s >= -1e-15]
            if not pos_list:
                print("No positive-strain data even though direction == "
                      "tension for %s" % model_name)
                return None
            truncated_pairs = []
            idx_ge = None
            for i, (s, st) in enumerate(pos_list):
                if s >= Threshold - 1e-15:
                    idx_ge = i
                    break
            if idx_ge is None:
                truncated_pairs = [(abs(s), st) for (s, st) in pos_list]
                last_s, last_st = pos_list[-1]
                if last_s != Threshold:
                    truncated_pairs.append((Threshold, last_st))
            else:
                if idx_ge == 0:
                    s_cur, st_cur = pos_list[0]
                    if abs(s_cur) < 1e-15:
                        sigma_t = st_cur
                    else:
                        sigma_t = (st_cur * (Threshold / s_cur)
                                   if s_cur != 0 else st_cur)
                    truncated_pairs.append((Threshold, sigma_t))
                else:
                    for j in range(0, idx_ge):
                        s_j, st_j = pos_list[j]
                        truncated_pairs.append((abs(s_j), st_j))
                    s_prev, st_prev = pos_list[idx_ge - 1]
                    s_cur, st_cur = pos_list[idx_ge]
                    if abs(s_cur - s_prev) < 1e-15:
                        sigma_t = st_cur
                    else:
                        t = (Threshold - s_prev) / (s_cur - s_prev)
                        sigma_t = st_prev + t * (st_cur - st_prev)
                    truncated_pairs.append((Threshold, sigma_t))
        else:
            if extreme_mag < Threshold - 1e-15:
                print("Max compression magnitude (%.6g) < Threshold (%.6g): "
                      "skipping CSV generation for model %s"
                      % (extreme_mag, Threshold, model_name))
                return None
            neg_list = [(abs(s), st) for (s, st) in sorted_signed if s <= 1e-15]
            if not neg_list:
                print("No negative-strain data even though direction == "
                      "compression for %s" % model_name)
                return None
            neg_list_sorted = sorted(neg_list, key=lambda t: t[0])
            idx_ge = None
            for i, (abs_s, st) in enumerate(neg_list_sorted):
                if abs_s >= Threshold - 1e-15:
                    idx_ge = i
                    break
            if idx_ge is None:
                truncated_pairs = [(abs_s, st)
                                   for (abs_s, st) in neg_list_sorted]
                last_s, last_st = neg_list_sorted[-1]
                if last_s != Threshold:
                    truncated_pairs.append((Threshold, last_st))
            else:
                if idx_ge == 0:
                    s_cur, st_cur = neg_list_sorted[0]
                    if abs(s_cur) < 1e-15:
                        sigma_t = st_cur
                    else:
                        sigma_t = (st_cur * (Threshold / s_cur)
                                   if s_cur != 0 else st_cur)
                    truncated_pairs = [(Threshold, sigma_t)]
                else:
                    truncated_pairs = []
                    for j in range(0, idx_ge):
                        truncated_pairs.append(
                            (neg_list_sorted[j][0], neg_list_sorted[j][1]))
                    s_prev, st_prev = neg_list_sorted[idx_ge - 1]
                    s_cur, st_cur = neg_list_sorted[idx_ge]
                    if abs(s_cur - s_prev) < 1e-15:
                        sigma_t = st_cur
                    else:
                        t = (Threshold - s_prev) / (s_cur - s_prev)
                        sigma_t = st_prev + t * (st_cur - st_prev)
                    truncated_pairs.append((Threshold, sigma_t))

    if not truncated_pairs:
        print("No truncated_pairs to process for %s" % model_name)
        return None

    energy_density = [0.0]
    for i in range(1, len(truncated_pairs)):
        eps_prev, sigma_prev = truncated_pairs[i - 1]
        eps_cur, sigma_cur   = truncated_pairs[i]
        delta_eps = eps_cur - eps_prev
        if delta_eps <= 0:
            area = 0.0
        else:
            area = 0.5 * (sigma_prev + sigma_cur) * delta_eps
        energy_density.append(energy_density[-1] + area)

    if sea_factor is not None:
        sea_list = [ed * sea_factor for ed in energy_density]
    else:
        sea_list = [0.0] * len(energy_density)
        print("WARNING: SEA set to 0 — missing mass info for %s" % model_name)

    if sea_list:
        print("Final SEA at max strain = %.6g N·mm/tonne = %.6g J/kg"
              % (sea_list[-1], sea_list[-1] * SEA_TO_J_KG))
    # ----------------------------------------------------------------

    csv_path = save_stress_strain_csv(
        model_name, truncated_pairs, energy_density,
        sea_list, data_dir)                                 

    return {'stress_csv': csv_path, 'n': len(truncated_pairs),
            'SEA_final_J_kg': sea_list[-1] * SEA_TO_J_KG if sea_list else None}


# ========================= 后处理 =========================

def post_process(model_name, odb_path, node_set='TOP'):
    if not os.path.exists(odb_path):
        print("ODB file not found: %s" % odb_path)
        return None

    for var in VARIABLES:
        if var in session.xyDataObjects:
            try:
                del session.xyDataObjects[var]
                print("Cleared residual session.xyDataObjects['%s']" % var)
            except Exception as e:
                print("Warning: failed to clear xyDataObjects['%s']: %s"
                      % (var, e))

    try:
        odb = session.openOdb(name=odb_path, readOnly=True)
    except Exception as e:
        print("Failed to open ODB %s: %s" % (odb_path, e))
        return None

    try:
        if not odb.steps:
            print("No steps found in ODB: %s" % odb_path)
            odb.close()
            return None
        first_step = list(odb.steps.values())[0]
        if not first_step.frames:
            print("No frames found in step: %s" % first_step.name)
            odb.close()
            return None
        print("ODB contains %d steps with data" % len(odb.steps))
    except Exception as e:
        print("Error checking ODB data: %s" % e)
        odb.close()
        return None

    try:
        viewport = session.viewports['Viewport: 1']
        viewport.setValues(displayedObject=odb)
        viewport.makeCurrent()
        viewport.odbDisplay.commonOptions.setValues(
            deformationScaling=UNIFORM, uniformScaleFactor=1.0)
        viewport.odbDisplay.basicOptions.setValues(
            renderBeamProfiles=ON, beamScale=1.0)
        viewport.odbDisplay.update()
    except Exception as e:
        print("Warning: Failed to set viewport options: %s" % e)

    for var in VARIABLES:
        try:
            if var == 'RF2':
                xyList = xyPlot.xyDataListFromField(
                    odb=odb, outputPosition=NODAL,
                    variable=(('RF', NODAL, ((COMPONENT, 'RF2'),)),),
                    nodeSets=(node_set,))
            else:
                xyList = xyPlot.xyDataListFromField(
                    odb=odb, outputPosition=NODAL,
                    variable=(('U', NODAL, ((COMPONENT, 'U2'),)),),
                    nodeSets=(node_set,))
        except Exception as e:
            print("Error extracting %s data: %s" % (var, e))
            xyList = None

        if not xyList:
            print("No %s data extracted" % var)
            continue

        if var == 'RF2':
            try:
                summed_curve = sum(tuple(xyList))
                if 'RF2' in session.xyDataObjects:
                    del session.xyDataObjects['RF2']
                session.xyDataObjects.changeKey(
                    fromName=summed_curve.name, toName='RF2')
            except Exception as e:
                print("Error processing RF2 data: %s" % e)
        else:
            try:
                avg_curve = avg(tuple(xyList))
                if 'U2' in session.xyDataObjects:
                    del session.xyDataObjects['U2']
                session.xyDataObjects.changeKey(
                    fromName=avg_curve.name, toName='U2')
            except Exception as e:
                print("Error processing U2 data: %s" % e)

    data_dir = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    info = convert_xydata_to_stress_strain_and_save(
        model_name, data_dir, thickness_override=None)
    print("Post-process conversion info: %s" % info)

    try:
        odb.close()
    except Exception:
        pass

    return info


def verify_beam_display(model_name):
    try:
        mdl = mdb.models[model_name]
        p = mdl.parts['PART-1']
        print("=== 梁显示验证 ===")
        print("模型: %s" % model_name)
        if hasattr(p, 'sectionAssignments'):
            for assignment in p.sectionAssignments:
                print("截面分配: %s" % assignment.sectionName)
        beam_orientations = p.beamOrientations
        if beam_orientations:
            for orient in beam_orientations:
                print("梁方向方法: %s" % orient.method)
                print("n1向量: %s" % str(orient.n1))
        print("==================")
    except Exception as e:
        print("梁显示验证失败: %s" % e)


# ========================= 主流程 =========================

if __name__ == '__main__':
    for fname in os.listdir(GRAPH_DIR):
        if not fname.lower().endswith('.inp'):
            continue

        inp_path  = os.path.join(GRAPH_DIR, fname)
        base_name = os.path.splitext(fname)[0]

        model_y = base_name
        job_y   = model_y
        odb_y   = os.path.join(BASE_DIR, '%s.odb' % job_y)

        print("\n" + "=" * 50)
        print("Processing Y-job for model: %s" % model_y)
        print("=" * 50)

        try:
            if model_y in mdb.models:
                del mdb.models[model_y]

            if os.path.exists(odb_y):
                try:
                    os.remove(odb_y)
                    print("Removed old ODB: %s" % odb_y)
                except Exception as e:
                    print("Warning: could not remove old ODB %s: %s"
                          % (odb_y, e))

            import_model(inp_path, model_y)
            define_material_and_section(model_y)
            verify_beam_display(model_y)
            assemble_and_step(model_y)

            success = submit_job(model_y, job_y)

            if success and os.path.exists(odb_y):
                print("Starting post-processing for %s" % model_y)
                post_process(model_y, odb_y, node_set='TOP')
            else:
                print("Skipping post-processing for %s — job failed or "
                      "ODB not found" % model_y)

        except Exception as e:
            print("Error processing %s: %s" % (model_y, e))
            continue

    print("\nAll models processed. Individual CSV files are in: %s"
          % os.path.join(BASE_DIR, 'data'))
    os._exit(0)