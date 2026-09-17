**π0.5 prefill 상위 GEMM의 Nsight Compute 측정 — deep9, 2026-09-17**

sudo를 통한 실제 GPU 성능 카운터 수집을 확인하고, batch 1·5에서 선택한 GEMM kernel 총 12개를 측정했다. B5의 두 kernel 종류에서 Tensor(HMMA) 파이프 지표가 약 **91.5%·94.0%**, DRAM 지표는 **36.6%·33.2%**였다. 이 표본에서는 DRAM 대역폭 포화보다 Tensor 연산 처리 쪽 제약을 우선 검토할 근거가 있다. 이 결과를 모든 prefill kernel 또는 모델 전체의 compute-bound 판정으로 확대하지 않는다.

**권한과 공용 서버 설정**

NCU 2026.1.0.0, driver 580.159.04, RTX A6000 GPU 0을 사용했다. GPU 1은 사용하지 않았다. `RmProfilingAdminOnly: 1`은 유지했다. 드라이버 모듈 재로드·재부팅·권한 제한 해제·전력 한도·팬 설정 변경은 하지 않았다. `--clock-control none`을 명시해 NCU의 자동 클럭 제어도 껐다.

sudo 인증은 정상 작동했다. 최초 CAP_PERFMON-only 모델 실행은 `ERR_NVGPUCTRPERM`을 남기고 유효 kernel을 수집하지 못했다. 작은 CUDA kernel 하나를 direct sudo와 사용자 UID + 일시적 CAP_SYS_ADMIN으로 각각 측정해 카운터 접근과 결과 검증을 확인했다. 이후 모델은 **UID/GID 1006인 사용자 계정**으로 실행하면서 해당 프로세스와 자식에 CAP_SYS_ADMIN만 부여했다. `--no-new-privs`를 설정하고 capability bounding set도 제한했다. CAP_SYS_ADMIN은 범용 권한이며, 권한 자체가 NCU 전용인 것은 아니다. 파일에 영구 capability를 부여하거나 사용자 계정의 지속적인 권한 설정을 바꾸지는 않았다. 비밀번호는 명령문·스크립트·실험 산출물에 저장하지 않았다.

NVIDIA는 제한된 성능 카운터 접근에 sudo 또는 적절한 capability가 필요하다고 설명한다. [권한 안내](https://developer.nvidia.com/ERR_NVGPUCTRPERM)

**측정 대상과 기준 trace의 관계**

[앞선 Nsight Systems 실험](2026-09-17-static-stage-nsight.md)은 VLM embedding·prefill·action을 NVTX/CUDA correlation 및 HLO로 연결했다. 이 기준 trace에서 B5 prefill의 상위 GEMM 두 종류는 호출당 약 95.6ms·46.6ms, 합계 약 62%를 차지했다. NCU에서는 그 kernel 이름들을 후보로 사용했다. B1은 기준 trace의 상위 CUTLASS 두 종류를 후보로 지정했다.

성공한 수집에서는 NVTX 필터를 사용하지 않았다. 앞선 NVTX 필터 실행이 첫 capture 호출에서 정체되어 중단됐으며, 실패한 수치는 최종 분석에 섞지 않았다. 실제 수집은 **kernel 이름 필터 + 첫 일치 6개 제한 + 외부 GPU 감시**로 수행했다. 어떤 변경이 정체를 해결했는지 단일 원인으로 확정하지 않는다.

분류 근거는 다음과 같다.

- 수집한 kernel 종류들이 기준 Nsight trace의 같은 B에서 `vlm_prefill`에만 나타났음을 자동 검증했다.
- 현재 NCU 실행과 기준 Nsight 실행의 debug 정보 제외 StableHLO가 B1·B5 모두 완전히 일치했다. annotation 전후의 일치도 확인했다.
- **현재 NCU 보고서에서 live NVTX stage 범위를 직접 수집한 것은 아니다.** 기준 trace의 분류를 같은 연산 그래프와 kernel 종류에 연결한 후속 측정이다.
- 최종 B1 표본 6개는 CUTLASS `128x128_32x5` 한 종류였다. B5는 CUTLASS `256x128_32x3` 4개와 cuBLAS Ampere GEMM 2개였다. B별 kernel 종류·launch 설정이 다르므로 두 B의 kernel 시간 평균을 나눠 동일 연산의 batch scaling이라고 해석하지 않는다.

모델은 `pi05_libero`, JAX SYNC, denoising 10회, action horizon 10이며 입력은 앞선 실험의 고정 LIBERO snapshot과 seed 7이다. Python 3.11.16/JAX 0.5.3/Flax 0.10.2의 같은 환경을 사용했다. GPU 실행 코드는 `e45b4bb`다. 모델의 monolithic JIT와 기본 CUDA graph 설정을 유지했다.

**실행과 샘플링**

B별로 별도 프로세스를 띄워 30회 warmup, capture-off 호출 1회 후 profiler를 켰다. 기존 실행기의 첫 `phase=transition, index=0` 호출 안에서 조건에 일치하는 kernel 6개를 수집했다. `--launch-count 6 --kill no`이므로 이후에는 수집을 끝내고 남은 추론과 출력 검증을 완료했다. 스크립트의 `phase=profile`이라는 이름은 이 NCU 실험의 카운터 수집 위치를 뜻하지 않는다.

`SpeedOfLight`, `ComputeWorkloadAnalysis`, `MemoryWorkloadAnalysis`, `Occupancy`, `LaunchStats`를 수집했다. 개별 kernel은 11 pass로 replay됐고 `--cache-control all`로 각 pass 전에 캐시를 비웠다. 그러므로 원래 실행 중의 따뜻한 캐시 상태나 kernel 간 동시성을 재현하는 측정이 아니다. NCU의 replay·cache 제어는 [공식 profiling 안내](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)에 설명되어 있다.

**측정 결과: 선택한 kernel 한 번의 평균**

| 표본 그룹 | 수집 kernel 수 | Kernel ms | Tensor 파이프 | DRAM | L2 | DRAM GB/s | 실제 occupancy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B1 CUTLASS | 6 | 0.604 | 74.5% | 36.4% | 63.7% | 265.6 | 8.33% |
| B5 cuBLAS Ampere | 2 | 2.304 | 94.0% | 33.2% | 62.7% | 241.9 | 16.63% |
| B5 CUTLASS | 4 | 2.392 | 91.5% | 36.6% | 61.3% | 266.8 | 16.66% |

퍼센트는 각 NCU 지표의 정의와 피크 기준이다. Tensor 열은 `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed`이며, 모델이 이론 최대 FLOPS의 같은 비율을 달성했다는 뜻은 아니다. DRAM은 `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`, L2는 `lts__throughput.avg.pct_of_peak_sustained_elapsed`다. NCU의 포괄적인 “Memory Throughput”을 그대로 DRAM 대역폭으로 사용하지 않았다. GB/s는 10진 단위다. 모든 원본 metric 이름과 단위 변환은 검증 JSON에 있다.

B1의 occupancy는 8.33%로 낮지만 이 launch의 이론 occupancy도 8.33%였다. 블록당 shared memory 82,944B가 동시 상주 블록 수를 제한했다. B5는 실제 16.63–16.66%, 이론 16.67%였다. 따라서 낮은 occupancy 숫자만으로 비효율을 결론내릴 수 없다. 이 표본에서는 낮은 occupancy와 높은 Tensor 파이프 지표가 동시에 관찰됐다.

**해석 범위**

이번 결과는 선택한 큰 BF16 GEMM에서 Tensor 연산 파이프가 높은 수준으로 사용되고 있음을 보여준다. 메모리 접근 비용도 존재하지만 DRAM 포화가 이 표본의 주된 제한이라는 증거는 얻지 못했다. L2 지표가 약 61–64%인 점도 함께 보아야 한다. 다른 transpose/reduction/fusion/copy kernel과 전체 prefill의 시간 분포는 기존 Nsight Systems 자료를 기준으로 별도 검토해야 한다.

NCU가 포함된 첫 capture 호출의 wall time은 B1 약 34.2초, B5 약 37.4초였다. 이것은 11-pass replay·저장/복원·계측 비용을 포함하므로 **실제 추론 latency로 사용하지 않는다**. 각 호출의 카운터 kernel ms도 NCU의 격리·캐시·클럭 조건에 따른 값이다. 앞선 일반 실행 및 Nsight Systems 시간과 그대로 대체하거나 더하지 않는다.

해당 capture 호출의 외부 1Hz GPU 기록은 B1 34개, B5 37개였다. B1 온도 66–69°C, SM clock 1800–1890MHz; B5 온도 68–72°C, 1800–1875MHz였고 SW thermal slowdown은 관찰되지 않았다. 앞선 Nsight 실험(83–87°C)과 열 상태가 다르므로 절대 시간의 직접 비교에 주의해야 한다. 클럭을 고정하지 않아 NCU도 관련 경고를 출력했다.

표본은 필터에 처음 일치한 6개씩이며 모든 레이어·입력·상황의 대표성을 보장하지 않는다. 정확한 FLOP/byte roofline, stage 전체의 가중 평균, 최적 batch, 다른 GPU에서의 성능은 이 실험에서 확정하지 않았다. prefix 길이와 padding이 실제 연산량에 미치는 영향은 후속 질문으로 남긴다.

**검증과 산출물**

12개 kernel, 지표 단위와 유한 값, 11 replay passes, 두 실행의 출력 shape·finite·기준 출력 일치, Nsight 기준 StableHLO 일치 검사를 통과했다. Kernel 이름과 grid/block이 같은 표본끼리만 집계했다. 분석 코드의 Ruff 검사도 통과했다.

최종 GUI 파일:

- `/home/jyjeong/armory/output/ncu_prefill_20260917/b1_gemm/prefill.ncu-rep`
- `/home/jyjeong/armory/output/ncu_prefill_20260917/b5_gemm/prefill.ncu-rep`

GUI에서 각 result의 `GPU Speed Of Light Throughput`, `Compute Workload Analysis`, `Memory Workload Analysis`, `Occupancy`를 확인하면 된다. 같은 kernel 이름이라도 grid/block 및 수집 조건을 함께 확인해야 한다.

공유용 [요약 CSV](data/2026-09-17-ncu-prefill-summary.csv), [kernel별 CSV](data/2026-09-17-ncu-prefill-kernels.csv), [검증 JSON](data/2026-09-17-ncu-prefill-validation.json), [수집 설정](data/2026-09-17-ncu-prefill-measurement-spec.json)을 보존했다. `.ncu-rep`, 전체 685개 열의 metric CSV, HLO, GPU 기록 및 실패한 시도는 `output/ncu_prefill_20260917/`에 보존하며 Git에 올리지 않는다.

재집계:

```bash
/usr/local/cuda-13.2/bin/ncu --import output/ncu_prefill_20260917/b1_gemm/prefill.ncu-rep \
  --page raw --csv --print-units base > output/ncu_prefill_20260917/b1_gemm/metrics.csv
/usr/local/cuda-13.2/bin/ncu --import output/ncu_prefill_20260917/b5_gemm/prefill.ncu-rep \
  --page raw --csv --print-units base > output/ncu_prefill_20260917/b5_gemm/metrics.csv
.venv/bin/python -m scripts.analyze_ncu_prefill output/ncu_prefill_20260917
```

GPU 측정 재현 시에는 수집 설정 JSON의 kernel 필터와 flags를 사용하고, 기존 `scripts.profile_static_stages`에 `--external-observer --batches 1` 또는 `5`, `--samples 1 --controls 1 --warmup 30`을 전달한다. `scripts.watch_profile_gpu`는 해당 새 출력 폴더를 대상으로 별도 프로세스에서 먼저 시작한다. GPU는 유휴 상태여야 하며 sudo/capability 실행은 서버에서 허용된 범위로 한정한다.

측정 종료 후 두 GPU 모두 메모리 0MiB·활동률 0%로 반환됐고, 상태 감시도 종료됐다. 일시적인 capability는 프로세스 종료와 함께 사라졌다.

후속 [VLM·action 구간 자원 사용률 측정](2026-09-17-stage-resources.md)에서 전체 GPU 활동 구간의 Tensor/SM 시계열과 NCU 메모리·warp·stall 지표를 추가 수집했다.
