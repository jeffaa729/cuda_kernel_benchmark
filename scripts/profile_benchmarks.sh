#!/usr/bin/env bash
set -euo pipefail

export PATH="/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin:${PATH:-}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

bench="${1:-all}"
if [[ $# -gt 0 ]]; then
    shift
fi

case "$bench" in
    -h|--help)
        usage
        exit 0
        ;;
esac

BIN="${BIN:-./build/bin/cuda_benchmarks}"
OUT_DIR="${OUT_DIR:-profiles}"
NCU="${NCU:-ncu}"
NCU_SET="${NCU_SET:-speed-of-light}"
NCU_METRICS="${NCU_METRICS:-gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active}"
VERBOSE="${VERBOSE:-0}"

mkdir -p "$OUT_DIR"

default_args_for() {
    case "$1" in
        vector_add) echo "1048576" ;;
        transpose) echo "1024" ;;
        reduction) echo "1048576" ;;
        gemm) echo "256" ;;
        softmax) echo "512 512" ;;
        conv2d) echo "8 8 32 32 16" ;;
        *) return 1 ;;
    esac
}

known_benchmark() {
    case "$1" in
        vector_add|transpose|reduction|gemm|softmax|conv2d) return 0 ;;
        *) return 1 ;;
    esac
}

profile_one() {
    local name="$1"
    shift
    local args=("$@")

    if [[ ! -x "$BIN" ]]; then
        echo "error: benchmark executable not found or not executable: $BIN" >&2
        echo "build first: cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel" >&2
        exit 1
    fi

    if ! command -v "$NCU" >/dev/null 2>&1; then
        echo "error: Nsight Compute CLI not found: $NCU" >&2
        echo "install it in WSL, or set NCU=/path/to/ncu" >&2
        exit 1
    fi

    if [[ ${#args[@]} -eq 0 ]]; then
        read -r -a args <<<"$(default_args_for "$name")"
    fi

    local base="${OUT_DIR}/${name}"
    local report="${base}.ncu-rep"
    local csv="${base}.csv"
    local summary="${base}_summary.txt"

    if [[ "$VERBOSE" == "1" ]]; then
        echo
        echo "== Profiling ${name} ${args[*]} =="
        echo "report:  $report"
        echo "csv:     $csv"
        echo "summary: $summary"
    fi

    local ncu_collection_args=()
    if [[ -n "$NCU_METRICS" ]]; then
        ncu_collection_args=(--metrics "$NCU_METRICS")
    else
        ncu_collection_args=(--set "$NCU_SET")
    fi

    "$NCU" \
        "${ncu_collection_args[@]}" \
        --target-processes all \
        --force-overwrite \
        -o "$base" \
        --csv \
        --page raw \
        --log-file "$csv" \
        "$BIN" "$name" "${args[@]}"

    summarize_csv "$name" "$csv" | tee "$summary"
}

summarize_csv() {
    local bench_name="$1"
    local csv_file="$2"

    awk -v bench="$bench_name" '
    function trim(s) {
        gsub(/^[ \t\r\n"]+|[ \t\r\n"]+$/, "", s)
        return s
    }

    function csvsplit(line, out,    i, c, q, field, n, nextc) {
        q = 0
        field = ""
        n = 1
        for (i = 1; i <= length(line); ++i) {
            c = substr(line, i, 1)
            nextc = substr(line, i + 1, 1)
            if (c == "\"" && q && nextc == "\"") {
                field = field "\""
                ++i
            } else if (c == "\"") {
                q = !q
            } else if (c == "," && !q) {
                out[n++] = field
                field = ""
            } else {
                field = field c
            }
        }
        out[n] = field
        return n
    }

    function approach(kernel, lower) {
        lower = tolower(kernel)
        if (lower ~ /vector_add.*naive/) return "Naive"
        if (lower ~ /transpose.*naive/) return "Naive"
        if (lower ~ /transpose.*shared/) return "Shared"
        if (lower ~ /transpose.*padding/) return "Padding"
        if (lower ~ /reduction.*interleave/) return "Interleave"
        if (lower ~ /reduction.*address/) return "Address"
        if (lower ~ /gemm.*naive/) return "Naive"
        if (lower ~ /gemm.*tiled/) return "Tiled"
        if (lower ~ /gemm.*register/) return "Register"
        if (lower ~ /cublas|sgemm|gemm/) return "CUBLAS/External"
        if (lower ~ /softmax.*naive/) return "Naive"
        if (lower ~ /softmax.*shared_memory/) return "SharedMemory"
        if (lower ~ /softmax.*warpshfl|softmax.*reg_cache/) return "WarpShuffleRegCache"
        if (lower ~ /conv2d.*naive/) return "Naive"
        if (lower ~ /conv2d.*tiled/) return "Tiled"
        return "Unknown"
    }

    function pretty_metric(metric) {
        if (metric == "gpu__time_duration.sum") return "runtime"
        if (metric == "sm__throughput.avg.pct_of_peak_sustained_elapsed") return "sm_pct"
        if (metric == "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed") return "mem_pct"
        if (metric == "dram__throughput.avg.pct_of_peak_sustained_elapsed") return "dram_pct"
        if (metric == "sm__warps_active.avg.pct_of_peak_sustained_active") return "occupancy_pct"
        return metric
    }

    function metric_value(key, metric,    value) {
        value = values[key SUBSEP metric]
        return value == "" ? "-" : value
    }

    function metric_unit(key, metric,    value) {
        value = units[key SUBSEP metric]
        return value == "" ? "" : value
    }

    BEGIN {
        printf "%-22s %-44s %14s %10s %10s %10s %10s %14s\n", bench, "Kernel", "Runtime", "Speedup", "SM %", "Mem %", "DRAM %", "Occupancy %"
        printf "%-22s %-44s %14s %10s %10s %10s %10s %14s\n", "--------", "------", "-------", "-------", "----", "-----", "------", "-----------"
    }

    /"Kernel Name"/ && !/"Metric Name"/ {
        delete header
        delete h
        fields = csvsplit($0, header)
        for (i = 1; i <= fields; ++i) {
            h[trim(header[i])] = i
        }
        mode = "wide"
        next
    }

    mode == "wide" {
        delete row
        fields = csvsplit($0, row)
        kernel = trim(row[h["Kernel Name"]])

        # Nsight Compute writes a units row immediately after the header.
        if (kernel == "" && trim(row[h["gpu__time_duration.sum"]]) == "ns") {
            units["wide" SUBSEP "runtime"] = trim(row[h["gpu__time_duration.sum"]])
            next
        }

        if (kernel == "") {
            next
        }

        key = kernel
        if (!(key in seen)) {
            seen[key] = 1
            order[++count] = key
        }

        values[key SUBSEP "runtime"] = trim(row[h["gpu__time_duration.sum"]])
        units[key SUBSEP "runtime"] = units["wide" SUBSEP "runtime"]
        values[key SUBSEP "sm_pct"] = trim(row[h["sm__throughput.avg.pct_of_peak_sustained_elapsed"]])
        values[key SUBSEP "mem_pct"] = trim(row[h["gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"]])
        values[key SUBSEP "dram_pct"] = trim(row[h["dram__throughput.avg.pct_of_peak_sustained_elapsed"]])
        values[key SUBSEP "occupancy_pct"] = trim(row[h["sm__warps_active.avg.pct_of_peak_sustained_active"]])
        next
    }

    /Metric Name/ {
        delete header
        delete h
        fields = csvsplit($0, header)
        for (i = 1; i <= fields; ++i) {
            h[trim(header[i])] = i
        }
        mode = "rows"
        next
    }

    mode == "rows" {
        delete row
        fields = csvsplit($0, row)
        kernel = trim(row[h["Kernel Name"]])
        metric = trim(row[h["Metric Name"]])
        unit = trim(row[h["Metric Unit"]])
        value = trim(row[h["Metric Value"]])

        if (kernel == "" || metric == "" || value == "") {
            next
        }

        key = kernel
        if (!(key in seen)) {
            seen[key] = 1
            order[++count] = key
        }

        metric_key = pretty_metric(metric)
        values[key SUBSEP metric_key] = value
        units[key SUBSEP metric_key] = unit
    }

    END {
        if (count == 0) {
            print "No kernel metrics found. Try setting NCU_SET=full or inspect the CSV directly:"
            print "  " FILENAME
            exit
        }

        naive_runtime_ns = 0
        for (i = 1; i <= count; ++i) {
            key = order[i]
            if (approach(key) == "Naive") {
                naive_runtime_ns = metric_value(key, "runtime") + 0
                break
            }
        }

        for (i = 1; i <= count; ++i) {
            key = order[i]
            runtime = metric_value(key, "runtime")
            runtime_ns = runtime + 0
            runtime_unit = metric_unit(key, "runtime")
            if (runtime != "-" && runtime_unit != "") {
                runtime = runtime " " runtime_unit
            }

            speedup = "-"
            if (naive_runtime_ns > 0 && runtime_ns > 0) {
                speedup = sprintf("%.2fx", naive_runtime_ns / runtime_ns)
            }

            display_kernel = key
            if (length(display_kernel) > 44) {
                display_kernel = substr(display_kernel, 1, 41) "..."
            }

            printf "%-22s %-44s %14s %10s %10s %10s %10s %14s\n",
                approach(key),
                display_kernel,
                runtime,
                speedup,
                metric_value(key, "sm_pct"),
                metric_value(key, "mem_pct"),
                metric_value(key, "dram_pct"),
                metric_value(key, "occupancy_pct")
        }
    }
    ' "$csv_file"
}

if [[ "$bench" == "all" ]]; then
    if [[ $# -gt 0 ]]; then
        echo "error: extra args are only valid when profiling one benchmark" >&2
        usage >&2
        exit 1
    fi
    first=1
    for name in vector_add transpose reduction gemm softmax conv2d; do
        if [[ "$first" == "0" ]]; then
            echo
        fi
        profile_one "$name"
        first=0
    done
elif [[ "$bench" == "summarize" ]]; then
    if [[ $# -ne 2 ]]; then
        echo "error: summarize needs a benchmark name and CSV path" >&2
        usage >&2
        exit 1
    fi
    summarize_csv "$1" "$2"
elif known_benchmark "$bench"; then
    profile_one "$bench" "$@"
else
    echo "error: unknown benchmark: $bench" >&2
    usage >&2
    exit 1
fi
