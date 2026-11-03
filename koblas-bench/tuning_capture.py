"""Read paired local captures or accepted rows from a fetched lab job."""
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from itertools import product


def sections(text):
    result, current = {"": {}}, ""
    for line in text.splitlines():
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            result.setdefault(current, {})
        elif "=" in line:
            key, value = line.split("=", 1)
            result[current][key] = value
    return result


def measured_manifest(path):
    manifest = path.parent / "source-hashes.txt"
    selected = {}
    for line in manifest.read_text().splitlines():
        digest, name = line.split(" ", 1)
        if re.match(r"(?:koblas/src/(?:[^/]*Main|jvmJava25|engineProfiles)/|koblas-bench/src/[^/]*Main/|koblas-bench/jmh/|gradle/)", name) or name in {
            "build.gradle.kts", "settings.gradle.kts", "gradle.properties", "koblas/build.gradle.kts", "koblas-bench/build.gradle.kts",
            "koblas-bench/capture-report.sh", "koblas-bench/script-helpers.sh", "koblas-bench/lab/vendor-capture.sh", "koblas-bench/lab/prepare-vendor.sh",
        }:
            if not re.fullmatch(r"[0-9a-f]{64}", digest) or name in selected:
                raise ValueError("invalid or repeated measured source hash")
            selected[name] = digest
    if not selected:
        raise ValueError("capture has no measured-path source manifest")
    return hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest(), manifest


def lab_parameters(lab):
    if not lab:
        return {}
    experiment = lab.get("resources", {}).get("lab-job.json", {}).get("experiment", {})
    parameters = experiment.get("bench", {}).get("request", {}).get("parameters", {}).copy()
    grid = experiment.get("grid") or {}
    arms = [{**(experiment.get("base") or {}), **config, **dict(zip(grid, values))}
            for config in experiment.get("configs") or [{}] for values in product(*grid.values())]
    varying = [key for key in dict.fromkeys(key for arm in arms for key in arm)
               if key != "label" and len({arm.get(key) for arm in arms}) > 1]
    selected = [arm for arm in arms if arm.get("label", " ".join(f"{key}={arm.get(key, '-')}" for key in varying) or "base") == lab.get("arm", "base")]
    if len(selected) != 1:
        raise ValueError("cannot recover the selected lab arm parameters")
    parameters.update({key.removeprefix("param."): value for key, value in selected[0].items() if key.startswith("param.")})
    return parameters


def protocol_signature(fields, identities, path, parameters):
    keys = ("warmups", "requested_samples", "target_ns", "requested_jvm_forks", "jvm_gc", "jvm_heap", "cpu_policy", "koblas_threads")
    if any(not fields.get(key) for key in keys) or not fields.get("platform"):
        raise ValueError("capture is missing timing or thread protocol metadata")
    bindings, mkl = [], False
    for identity in identities:
        if not identity.get("runtime") or not identity.get("harness") or not identity.get("warmup_target_ns"):
            raise ValueError("capture is missing runtime or harness metadata")
        vendors = re.findall(r"/(Accelerate|oneMKL|OpenBLAS|AOCL|ArmPL)/(.*?)/threads=([^/]+)", identity.get("implementation", ""))
        mkl |= any(vendor == "oneMKL" for vendor, _, _ in vendors)
        if not vendors and set(re.findall(r"/caps\[([^:]+):", identity.get("implementation", ""))) & {"onemkl", "openblas", "accelerate", "aocl", "armpl"}:
            raise ValueError("capture is missing vendor version and thread evidence")
        bindings.append([identity["runtime"], identity["harness"], identity["warmup_target_ns"],
                         [(vendor, location.rsplit("/", 1)[-1], threads) for vendor, location, threads in vendors]])
    package = path.parent.parent / "vendor-package.txt"
    if mkl and parameters.get("vendor") == "onemkl" and identities[0].get("profile_os") == "Linux" and not package.is_file():
        raise ValueError("lab oneMKL capture is missing its pinned package evidence")
    protocol = {key: value for key, value in fields.items() if key in keys or key.startswith(("KOBLAS_DENSE_", "KOBLAS_SPARSE_")) or key == "engine_profile_sha256"}
    if any(identity.get("profile_runtime") == "jvm" for identity in identities):
        protocol.update(JAVA_TOOL_OPTIONS=fields.get("JAVA_TOOL_OPTIONS", parameters.get("jvm-options", "")),
                        JDK_JAVA_OPTIONS=fields.get("JDK_JAVA_OPTIONS", ""),
                        KOBLAS_BENCH_PROFILE_GC=fields.get("KOBLAS_BENCH_PROFILE_GC", parameters.get("profile-gc", "false")))
    return dict(protocol=protocol, bindings=bindings, system=fields["platform"].split()[2:],
                package=package.read_text().strip() if mkl and package.is_file() else None)


def instruction_class(identities, hardware, lab):
    declared = {identity.get("profile_cpu_class", "*") for identity in identities}
    if len(declared) != 1:
        raise ValueError("paired arms report different CPU instruction classes")
    value = next(iter(declared))
    if value != "*":
        return value
    flags = set(hardware.get("flags", "").split())
    platform = identities[0]
    required = lab.get("resources", {}).get("lab-job.json", {}).get("experiment", {}).get("nodeTags") or {}
    return ("Sve" if platform.get("profile_arch") == "Arm64" and "sve" in flags else
            "Arm64" if platform.get("profile_arch") == "Arm64" and (platform.get("profile_os") == "MacOs" or flags & {"asimd", "neon"}) else
            "Avx512" if "avx512f" in flags or "simd:avx512" in required.get("require", []) else
            "Avx2" if "avx2" in flags else "X86_64" if flags and platform.get("profile_arch") == "X86_64" else "*")


def capture_groups(directory, arm, targets):
    index = directory / "lab-cases.jsonl"
    if not index.exists():
        if arm:
            raise ValueError("--lab-arm requires a fetched job")
        return [sorted(p for p in directory.glob("pass-*") if p.is_dir())], {}
    rows = [json.loads(line) for line in index.read_text().splitlines() if line.strip()]
    if any(type(row["index"]) is not int or row["index"] < 0 for row in rows) or len({row["index"] for row in rows}) != len(rows):
        raise ValueError("invalid or repeated lab case index")
    arms = {row["arm"] for row in rows}
    if arm is None and len(arms) == 1:
        arm = next(iter(arms))
    if arm not in arms:
        raise ValueError(f"select --lab-arm from {sorted(arms)}")
    paired = {}
    for row in rows:
        key = (row["problem"]["problem"], row["seed"], row["repeat"])
        paired.setdefault(key, []).append(row)
    groups, accepted = [], []
    for members in paired.values():
        if {row["arm"] for row in members} != arms or len(members) != len(arms) or any(
            row["status"] != "DONE" or (row.get("record") or {}).get("status") != "success"
            for row in members
        ):
            continue
        row = next(row for row in members if row["arm"] == arm)
        root = (directory / "cases" / str(row["index"]) / "out").resolve()
        artifacts = {artifact["path"] for artifact in row["record"]["artifacts"]}
        def declared_path(name):
            path = (root / name).resolve()
            if name not in artifacts or not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f"invalid or missing accepted artifact: {path}")
            return path
        paths = []
        for artifact in row["record"]["artifacts"]:
            name = artifact["path"]
            if not re.search(r"(?:^|/)pass-\d+/metadata\.txt$", name):
                continue
            path = declared_path(name)
            if any(name.removesuffix("metadata.txt") + target + ".csv" not in artifacts for target in targets):
                raise ValueError("accepted row does not declare every paired CSV artifact")
            declared_path(str((path.parent.parent / "source-hashes.txt").relative_to(root)))
            package = path.parent.parent.parent / "vendor-package.txt"
            if package.is_file():
                declared_path(str(package.relative_to(root)))
            paths.append(path.parent)
        if len(paths) != len(set(paths)) or not paths:
            raise ValueError("accepted row has missing or repeated capture rounds")
        groups.append(sorted(paths))
        accepted.append(row["index"])
    if not groups:
        raise ValueError("no rows where every arm completed successfully")
    # Execution snapshots retain preparation and failed-attempt resource counters as well as winners.
    resources = {name: json.loads((directory / name).read_text())
                 for name in ("lab-job.json", "lab-execution.json")}
    if resources["lab-job.json"]["status"] != "DONE" or type(resources["lab-job.json"].get("id")) is not int:
        raise ValueError("fitting requires a completed job snapshot")
    return groups, dict(arm=arm, accepted_rows=accepted, rows=rows, resources=resources,
                        requested=[key for key in paired], accepted=[key for key, members in paired.items()
                        if any(row["index"] in accepted for row in members)])


@dataclass
class Capture:
    paired: dict
    hashes: dict
    shapes: dict
    providers: dict
    by_cpu: dict
    platforms: set
    models: set
    commits: set
    cache_by_cpu: dict
    lab: list
    classes: set
    provenance: list = field(default_factory=list)
    missing_rows: set = field(default_factory=set)
    modes: dict = field(default_factory=dict)
    sve_widths: set = field(default_factory=set)
    excluded: list = field(default_factory=list)

    def target(self, runtime, operating_system=None, architecture=None, cpu=None):
        if len(self.platforms) != 1:
            raise ValueError("fitting requires a single runtime, OS and architecture")
        measured = next(iter(self.platforms))
        requested = (runtime, operating_system, architecture)
        if any(want is not None and want != actual for want, actual in zip(requested, measured)):
            raise ValueError(f"requested target {requested} differs from measured target {measured}")
        if cpu != "*" and (len(self.models) != 1 or "*" in self.models or cpu is not None and cpu not in self.models):
            raise ValueError("CPU model is mixed, unknown or mismatched; select one model or explicitly use --cpu '*'")
        if len(self.classes) != 1:
            raise ValueError("CPU instruction classes differ; fit each target separately")
        cpu_class = next(iter(self.classes))
        if cpu == "*" and cpu_class == "*":
            raise ValueError("a model-independent profile requires a verified CPU instruction class")
        if cpu_class == "Sve" and (len(self.sve_widths) != 1 or "*" in self.sve_widths):
            raise ValueError("SVE fitting requires one verified vector width; recapture or select --sve-vector-bits")
        return (*measured, cpu or next(iter(self.models)), cpu_class, next(iter(self.sve_widths)) if cpu_class == "Sve" else "*")


def measurements(directories, baseline, candidate, backend=None, strict=False, arm=None, profile_sha256=None,
                 cpu_class=None, sve_vector_bits=None):
    if len({directory.resolve() for directory in directories}) != len(directories):
        raise ValueError("repeated capture directory")
    targets = ["jvm-scalar", "jvm-simd", candidate] if baseline == "jvm-owned" else [baseline, candidate]
    inputs = [capture_groups(directory, arm, targets) for directory in directories]
    jobs = [lab["resources"]["lab-job.json"].get("id") for _, lab in inputs if lab]
    if len(jobs) != len(set(jobs)):
        raise ValueError("repeated input lab job")
    result = Capture({}, {}, {}, {}, {}, set(), set(), set(), {}, [lab for _, lab in inputs], set())
    requested = {tuple(key) for _, lab in inputs for key in lab.get("requested", [])}
    result.missing_rows = requested - {tuple(key) for _, lab in inputs for key in lab.get("accepted", [])}
    if strict and result.missing_rows:
        raise ValueError("validation has requested rows without accepted results")
    protocol, environments = None, {}
    for passes, lab in ((passes, lab) for groups, lab in inputs for passes in groups):
        parameters = lab_parameters(lab)
        if len(passes) < 3:
            raise ValueError("at least three paired capture rounds are required in each accepted row")
        first = sections((passes[0] / "metadata.txt").read_text())
        selected_class = instruction_class([first.get(target, {}) for target in targets], first.get("hardware", {}), lab)
        selected_width = first.get(targets[0], {}).get("profile_sve_vector_bits", "*")
        if cpu_class and selected_class == "*" or sve_vector_bits and selected_class == "Sve" and selected_width == "*":
            raise ValueError("cannot filter a capture with unverified CPU class or SVE width")
        if cpu_class and selected_class != cpu_class or sve_vector_bits and selected_width != str(sve_vector_bits):
            result.excluded.append(dict(capture=str(passes[0].parent), cpu_class=selected_class, sve_vector_bits=selected_width,
                                        job=lab.get("resources", {}).get("lab-job.json", {}).get("id")))
            continue
        paired, models, commits, manifests, expected = {}, set(), set(), set(), None
        for path in passes:
            metadata = sections((path / "metadata.txt").read_text())
            fields = metadata.get("capture", metadata[""])
            if profile_sha256 is not None and fields.get("engine_profile_sha256") != profile_sha256:
                raise ValueError("validation capture does not match the requested profile hash")
            if metadata[""].get("status") != "complete" or int(fields.get("requested_samples", 0)) < 3 or int(fields.get("target_ns", 0)) < 10_000_000:
                raise ValueError(f"{path}: incomplete capture or smoke timing settings")
            commit = fields.get("source_commit", "")
            if not re.fullmatch(r"[0-9a-f]{40}", commit) or fields.get("dirty") != "false":
                raise ValueError(f"{path}: fitting requires a clean, recorded source commit")
            commits.add(commit)
            if len(commits) != 1:
                raise ValueError("capture rounds within one worker attempt use different source commits")
            result.commits.add(commit)
            identities = [metadata.get(target, {}) for target in targets]
            signature = protocol_signature(fields, identities, path, parameters)
            if protocol is not None and protocol != signature["protocol"]:
                raise ValueError("capture rows use different timing protocols")
            protocol = signature["protocol"]
            manifest, source_hashes = measured_manifest(path)
            manifests.add(manifest)
            result.hashes[str(source_hashes)] = hashlib.sha256(source_hashes.read_bytes()).hexdigest()
            if len(manifests) != 1:
                raise ValueError("capture rounds within one worker attempt use different measured code")
            owned = [set(re.findall(r"/caps\[([^:]+):", identity.get("implementation", ""))) for identity in identities[:2]]
            if baseline == "jvm-owned" and any(not sources or sources - {"portable", "jvm-simd"} for sources in owned):
                raise ValueError("owned baseline contains vendor providers")
            for identity in identities:
                platform = tuple(identity.get("profile_" + key) for key in ("runtime", "os", "arch"))
                model = identity.get("profile_cpu_model")
                if not all(platform) or not model:
                    raise ValueError(f"{path}: missing runtime platform identity; recapture with the current harness")
                if platform[0] == "jvm" and (int(fields.get("warmups", 0)) < 20 or
                    int(fields.get("target_ns", 0)) < 200_000_000 or int(fields.get("requested_jvm_forks", 0)) < 2):
                    raise ValueError("JVM fitting needs twenty 200 ms warmups and two forks")
                result.platforms.add(platform)
                result.models.add(model)
            if any(identities[0].get(key) != identity.get(key) for identity in identities[1:]
                   for key in ("profile_runtime", "profile_os", "profile_arch", "profile_cpu_model")):
                raise ValueError("paired arms report different runtime platform identities")
            hardware = metadata.get("hardware", {})
            if instruction_class(identities, hardware, lab) != selected_class:
                raise ValueError("capture rounds within one worker attempt report different CPU classes")
            widths = {identity.get("profile_sve_vector_bits", "*") for identity in identities}
            if selected_class == "Sve":
                if widths != {selected_width} or selected_width != "*" and (not selected_width.isdigit() or int(selected_width) not in range(128, 2049, 128)):
                    raise ValueError("invalid or inconsistent SVE vector width")
                result.sve_widths.update(widths)
            result.classes.add(selected_class)
            model = hardware.get("model name", hardware.get("cpu_model", identities[0]["profile_cpu_model"]))
            if model == "*":
                raise ValueError("missing CPU model for measurement stratification")
            environment = {key: value for key, value in signature.items() if key != "protocol"}
            if environments.setdefault(model, environment) != environment:
                raise ValueError("capture rows use different runtime vendor or OS protocols within a CPU model")
            models.add(model)
            cache = re.fullmatch(r"([\d.]+)\s*([KMGT]?)i?B(?:\s+.*)?", hardware.get("l3 cache", ""))
            capacity = int(float(cache[1]) * 1024 ** (" KMGT".index(cache[2]))) if cache else int(hardware.get("l2_cache_bytes", 0)) or None
            topology = [hardware.get(key, "0") for key in ("core(s) per socket", "socket(s)")]
            cores = math.prod(int(value) if value.isdecimal() else 0 for value in topology)
            if cache and cores > 0:
                capacity = (capacity + cores - 1) // cores
            if model in result.cache_by_cpu:
                previous = result.cache_by_cpu[model]
                capacity = max(previous or 0, capacity or 0) or None
            result.cache_by_cpu[model] = capacity
            exported = re.findall(r"/caps\[" + re.escape(backend or "") + r":([^]]*)\]", identities[-1].get("implementation", ""))
            supported = set(exported[-1].split(',')) if exported else None
            if backend and supported is None and (backend == "jvm-simd" or candidate.endswith("-default")):
                raise ValueError("combined fitting requires capability metadata")
            rows = []
            for target in targets:
                with (path / f"{target}.csv").open() as source:
                    table = list(csv.DictReader(source))
                keyed = {row["case"]: row for row in table}
                if len(keyed) != len(table):
                    raise ValueError("repeated cases in a capture round")
                rows.append(keyed)
            case_sets = [set(table) for table in rows]
            if expected is not None and expected != case_sets:
                raise ValueError("capture rounds contain different cases")
            expected = case_sets
            if baseline == "jvm-owned":
                if case_sets[0] != case_sets[1]:
                    raise ValueError("owned comparison arms contain different cases")
                best = {}
                for case in rows[0]:
                    variants = [table[case] for table in rows[:2]]
                    if any(row["status"] != "ok" for row in variants) or any(
                        variants[0][key] != variants[1][key] for key in ("timing_mode", "routing_key")
                    ) or not all(math.isfinite(float(row["median_ns"])) and float(row["median_ns"]) > 0 for row in variants):
                        raise ValueError(f"{case}: owned baseline is incomplete or incomparable")
                    best[case] = min(variants, key=lambda row: float(row["median_ns"]))
                rows = [best, rows[-1]]
            if strict and rows[0].keys() != rows[1].keys():
                raise ValueError("validation arms contain different cases")
            for case in rows[0].keys() & rows[1].keys():
                left, right = rows[0][case], rows[1][case]
                if result.modes.setdefault(case, left["timing_mode"]) != left["timing_mode"]:
                    raise ValueError(f"{case}: timing boundaries changed between capture inputs")
                if left["status"] != "ok" or right["status"] != "ok":
                    if strict or left["status"] != "ok" or supported and left["routing_key"].split(':')[0] in supported:
                        raise ValueError(f"{case}: validation contains an untimed case")
                    continue
                if any(left[key] != right[key] for key in ("timing_mode", "routing_key")):
                    raise ValueError(f"{case}: different timing boundaries or routing dimensions")
                key = left["routing_key"].split(":")
                if key[0] and (len(key) != 7 or any(key[i] not in ("true", "false") for i in (4, 6))):
                    raise ValueError(f"{case}: malformed routing dimensions")
                shape = (key[0], *map(int, key[1:4]), key[4] == "true", int(key[5]), key[6] == "true") if key[0] else None
                provider = right["provider"]
                if result.shapes.setdefault(case, shape) != shape:
                    raise ValueError(f"{case}: routing dimensions changed between rounds")
                if backend and provider != backend or supported is not None and shape and shape[0] not in supported:
                    continue
                if not provider or result.providers.setdefault(case, {}).setdefault(model, provider) != provider:
                    raise ValueError(f"{case}: missing or inconsistent selected provider")
                times = [float(row["median_ns"]) for row in (left, right)]
                if not all(math.isfinite(t) and t > 0 for t in times) or any(int(row["samples"]) < 3 for row in (left, right)):
                    raise ValueError(f"{case}: invalid timings or insufficient measured samples")
                paired.setdefault(case, []).append(times[1] / times[0])
            for name in ["metadata.txt", *(f"{target}.csv" for target in targets)]:
                result.hashes[str(path / name)] = hashlib.sha256((path / name).read_bytes()).hexdigest()
        if len(models) != 1:
            raise ValueError("capture rounds within one worker attempt report different CPU models")
        complete = {case: ratios for case, ratios in paired.items() if len(ratios) == len(passes)}
        if strict and (not complete or len(complete) != len(rows[0])):
            raise ValueError("validation requires complete paired timings for every case")
        result.provenance.append(dict(job=lab.get("resources", {}).get("lab-job.json", {}).get("id"),
                                      capture=str(passes[0].parent),
                                      commit=next(iter(commits)), manifest=next(iter(manifests)),
                                      signature=signature, cpu=next(iter(models)), paired=complete))
        for case, ratios in complete.items():
            result.paired.setdefault(case, []).extend(ratios)
            result.by_cpu.setdefault(case, {}).setdefault(next(iter(models)), []).extend(ratios)
    if len({record["manifest"] for record in result.provenance if record["paired"]}) > 1:
        raise ValueError("profile inputs use different measured sources; recapture on one revision")
    return result
