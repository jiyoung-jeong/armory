**Armory 고정 입력 batch·4대 실행·Nsight 후속 실험 — 2026-09-15, deep9**

후속 [mirror 기준점 검증·4대 스케줄러 비교](2026-09-15-mirror-and-scheduler.md)에서 정합성 수정 후 9회를 새로 실행했다. 코드가 달라 이 보고서의 6회와 합산하지 않는다.

고정 입력 추론에서는 batch를 키울수록 요청당 비용이 줄었지만, 로봇 4대의 실행에서는 최대 batch 2가 4보다 action 공급 연속성과 성공 작업 수에서 나았다. 이번 조건에서 최대 batch 4가 생성한 chunk는 더 많았으나, 이미 지난 prefix와 이후 교체되는 action이 늘어 실제 실행 action은 감소했다. 따라서 다음 최적화의 평가 기준은 raw chunk 처리량뿐 아니라 **로봇별 queue에 순증가한 action, 실제 사용 action, 부족 구간의 길이**를 포함해야 한다.

이 보고서는 이전 reset 경계 문제를 수정한 뒤 새로 수행한 기준 실행 6회, 고정 입력 측정 1,100회, 4대의 Nsight 분석을 다룬다. 기존 2026-09-14 실행은 이전 에피소드의 응답이 다음 에피소드에 적용되는 문제가 포함되어 있으므로 새 기준 실행과 합산하지 않았다. 이전 분석은 [chunk 사용량 보고서](2026-09-15-chunk-usage.md)에 보존한다.

**조건과 측정 범위**

- GPU 0: RTX A6000 48GB, UUID `GPU-48f8ed9c-ab6c-5134-2277-bd7e0f487b4b`, 드라이버 580.159.04. 시작 전 사용 여부를 검사하고 실행 중 다른 compute 프로세스 유입을 확인했다. GPU 1은 사용하지 않았다.
- Python 3.11.16, JAX 0.5.3, PyTorch 2.7.1/cu126, MuJoCo 3.7.0, robosuite 1.4.1. 사용자 `.venv`와 캐시만 사용했다.
- 모델 `pi05_libero`, 로컬 checkpoint `.cache/openpi-private/cache/openpi-assets/checkpoints/pi05_libero`, denoising 10 step. 실제 요청은 **SYNC**다. warmup에 RTC shape가 포함되지만 이 실험을 RTC 실행으로 해석하면 안 된다.
- action horizon 10, 반환 shape 10×7, 모델 내부 action dimension 32. 4대 모두 20 Hz, Hmin=1/Hmax=10, LIBERO-10 task `[5,2,6,9]`, seed `[7,8,9,10]`, max episode step 500.
- scheduler `lookahead-actions`, alpha=1, search depth=1, max inflight=1, step budget=8, 동일 가중치. 로봇별 task와 seed는 고정하고 시간 제한 동안 episode를 반복했다. 서로 다른 무작위 seed에 대한 일반화 실험은 아니다.
- loopback `127.0.0.1:8080`; 실제 rollout에서는 정책 추론과 LIBERO EGL 렌더링이 GPU 0을 공유했다. 시스템 설정, 드라이버, Python 기본 경로, GPU clock·power·fan은 변경하지 않았다.

**먼저 수정하고 확인한 episode 경계**

reset마다 새 episode UUID를 발급하고 request/reset/response/ACK 및 engine slot에 전달했다. 서버와 클라이언트가 이전 세대의 응답을 거부하며, 클라이언트 검사는 reset과 같은 lock 안에서 수행한다. 이미 수신한 응답이 lock을 기다리는 동안 reset되는 경우도 회귀 검증했다. 서버는 전송 시작·완료와 폐기 사유를, 클라이언트는 reset, step, chunk 수신 직전·직후 queue와 packet 처리 시간을 선택적으로 기록한다.

새 기준 실행 6회의 **305개 episode, 84,497개 기록 step, 9,261개 저장 chunk**를 대조했다. 모든 기록 action의 출처와 pop 직전 queue 길이가 재구성값 및 broker 이벤트에 일치했고, 다른 episode 응답의 저장·적용은 0건이다. 이는 이번 실행에 대한 검증 결과이며 모든 가능한 경쟁 상태가 없다는 증명은 아니다.

| 응답 경로, 6회 합계 | 개수 |
| --- | ---: |
| 서버 처리 | 9,643 |
| 서버 전송 완료 | 9,488 |
| 클라이언트 수용 = ACK | 9,483 |
| episode 결과에 저장 | 9,261 |
| 저장 후 기록 step에서 1회 이상 사용 | 9,234 |
| 서버에서 이전 세대 폐기 | 155 |
| 클라이언트에서 이전 세대 폐기 | 5 |

처리했지만 저장되지 않은 382개는 수용 후 snapshot에 없는 222개와 폐기한 160개로 모두 설명됐다. 222개는 원 episode의 마지막 기록 step 이후에 수신됐으며, 완료 episode 206개·시간 제한으로 중단된 episode 16개다. snapshot 시각 자체는 별도로 계측하지 않았다. 저장 chunk 중 한 번도 실행되지 않은 27개는 마지막 snapshot의 queue에 남아 있었다. 전송 경로를 설명하지 못한 응답은 0개다.

**4대 기준 실행: 프로파일러 없이 180초씩 3회**

조건 순서는 B2/B4, B4/B2, B2/B4로 바꿨다. 매회 새 서버를 시작하고 warmup은 180초에 포함하지 않았다. 아래는 실행별 통계의 3회 평균이다. ±는 실행 간 표본 표준편차이며 신뢰구간이 아니다.

| 지표 | 최대 batch 2 | 최대 batch 4 |
| --- | ---: | ---: |
| 성공 작업/분 | 15.56 ± 0.19 | 14.67 ± 0.00 |
| 첫 모델 action 이후 부족 step 비율 | 19.90% ± 0.50%p | 21.68% ± 0.87%p |
| chunk 지연 p95, ms | 276.72 | 335.58 |
| 실제 batch 크기 평균 | 1.968 | 2.068 |
| chunk 수신 간격 평균, ms | 461.86 | 442.09 |
| 서버 처리 chunk/s | 8.741 | 9.117 |
| 기록된 모델 action/s | 60.909 | 59.587 |
| chunk당 실행 action 평균 | 7.267 | 6.797 |
| chunk당 queue 순증가 평균 | 7.371 | 6.901 |
| 샘플링한 GPU 메모리 최대, MiB | 11,138 | 11,192 |

chunk 지연은 저장 chunk의 요청 timestamp→broker 수용 시각이며 관측 생성 시간은 빠진다. 전체 sensor-to-action age가 아니다. queue의 `actions_left`는 step의 pop 이전 길이다. 부족 비율은 첫 모델 action 실행 이후 action 없이 null action을 사용한 **기록 step의 비율**이지 전체 wall time 비율이 아니다. chunk gap은 수신 간격이며, packet을 서버에 보내는 간격과 다르다.

최대 batch 4는 매번 4개를 묶는 설정이 아니다. 실제 평균은 약 2.07이었다. 예를 들어 B4 첫 실행의 795개 batch 중 크기 4는 17개였다. 이 실행에서 모든 scheduler 탐색이 완료됐고 탐색한 후보 수 1/2/3/4의 빈도는 172/341/250/32였다. 동일 가중치에서는 크기별 EDF prefix 후보를 생성하므로 최대 4개 후보였고 step budget 8 고갈로 결과가 제한된 것은 아니다. 이 관측만으로 다른 목적 함수의 개선량을 단정하지는 않는다.

| 저장 action의 분류, 각 조건 3회 합계 | 최대 batch 2 | 최대 batch 4 |
| --- | ---: | ---: |
| 저장 chunk × 10 | 45,260 | 47,350 |
| 기록 step에서 실행 | 32,891 | 32,177 |
| 도착 시 이미 지난 prefix | 11,628 | 13,468 |
| 후속 chunk로 교체 | 269 | 1,211 |
| 마지막 snapshot에 남음 | 472 | 494 |

네 분류의 합은 모든 chunk에서 정확히 10개다. B4는 저장 action을 2,090개 더 만들었지만 prefix가 1,840개, 교체가 942개, 잔여가 22개 늘어 **실제로 실행한 action은 714개 감소**했다. 정상적인 미래 action 갱신도 포함되므로 교체를 모두 낭비라고 부르지 않는다. 수량 분해는 이 실행의 공급 차이를 설명하지만 성공률 차이의 인과관계 전체를 증명하지 않는다.

부족 구간은 B2 3,259회, B4 3,355회였다. 길이 p95는 4→5 step, 최장은 8→7 step이다. 최장 구간 하나는 짧아졌으나 전체 부족 비율과 p95는 나빠졌다. 6,614개 부족 시작 중 6,471개(97.84%)가 해당 로봇을 회복시키는 다음 chunk의 `infer_batch` 실행 중에 발생했다. 이것은 함수 wall time과의 겹침이며 그 순간 GPU kernel이 실행 중이었다는 뜻은 아니다.

**실제 관측을 고정한 batch 비용**

LIBERO task 4개에서 얻은 초기 실제 관측을 각각 NPZ로 고정했다. 각 관측은 state 8, 두 카메라 224×224×3 uint8, task prompt를 포함한다. 각 파일의 SHA256은 `output/fixed_inputs_20260915/metadata.json`에 있다. 크기 B에는 동일한 관측 목록의 앞 B개를 사용했다. 따라서 같은 B의 입력은 반복마다 같고, 서로 다른 B에는 서로 다른 개수의 실제 관측이 포함된다.

관측 생성 프로세스를 종료해 EGL을 제거한 후 같은 checkpoint·추론 함수로 측정했다. SYNC/RTC shape 1~4 warmup 후 실제 고정 입력으로 shape별 5회 더 warmup했다. 측정 호출마다 RNG key 7을 timer 밖에서 복원하고 모델의 정상 noise 생성 경로를 사용했다. `infer_batch`가 반환하는 NumPy action/state/noise까지 물질화된 결과를 받는 wall time이므로 GPU 완료를 기다린다. 순수 kernel 시간만 측정한 수치는 아니다. 서버·scheduler·직렬화·네트워크·렌더러는 fixed/dynamic 단계에 없다.

각 B를 50회씩 3 block, 총 600회 측정했다. block마다 B 순서를 회전했다.

| 고정 B | 횟수 | 평균 ms | p50 ms | p95 ms | 요청당 ms | 요청/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 150 | 136.25 | 134.40 | 138.42 | 136.25 | 7.340 |
| 2 | 150 | 202.62 | 202.70 | 203.94 | 101.31 | 9.870 |
| 3 | 150 | 255.84 | 255.78 | 258.11 | 85.28 | 11.726 |
| 4 | 150 | 306.13 | 305.98 | 308.88 | 76.53 | 13.066 |

B4의 요청 처리율은 B1의 1.780배다. B1에서 367.44 ms인 한 호출이 있었고 원인을 별도 추적으로 확정할 자료는 없다. 임의로 제거하지 않고 평균·p95·최댓값과 원시 `calls.jsonl`에 유지했다.

미리 컴파일한 shape를 **1→2→3→2**로 바꾸는 200회 측정에서는 평균 B1=134.81 ms, B2=203.21 ms, B3=255.94 ms였다. 이 조건에서는 warm shape 변경 자체가 큰 추가 비용을 보이지 않았다. 새 shape의 최초 compilation 비용이나 모든 동적 실행 패턴까지 측정한 것은 아니다.

추가로 동일 모델·입력을 유지하면서 시뮬레이터 4개를 끈 상태→켠 상태→끈 상태로 비교했다. 각 단계 B2/B4 50회씩, 300회다. 시뮬레이터는 같은 GPU에서 null action·두 카메라·관측 전처리를 20 Hz로 실행했고 실제 속도는 약 19.8 Hz였다. WebSocket은 없다.

| B | 켜기 전 평균 ms | 4개 실행 중 평균 ms | 끈 뒤 평균 ms |
| --- | ---: | ---: | ---: |
| 2 | 203.70 | 218.30 | 203.78 |
| 4 | 306.66 | 325.15 | 306.82 |

추가 부하에서 약 7.1%/6.0% 증가하고 종료 후 돌아왔다. CPU 물리 시뮬레이션·전처리와 GPU 렌더링을 함께 추가한 실험이므로 **GPU 렌더링만의 비용**으로 해석할 수 없다. null action 부하도 실제 정책 rollout과 다르다. fixed 600 + dynamic 200 + control 300 = 1,100개 측정 호출 모두 출력 10×7·유한 값·동일 입력/RNG 결과 일치(atol/rtol 1e-5)를 확인했다.

**전송 정책에 대한 판단**

일반 실행의 관측 packet은 약 302 KB였고 직렬화 평균 약 0.16 ms, WebSocket send 호출 약 0.35 ms였다. 선택된 관측의 request→서버 arrival은 약 1.6~1.7 ms, 추론 종료→전송 시작 약 1.4~1.6 ms, send 완료→수신 약 0.6 ms, 수신→broker 수용 약 0.02 ms였다. 같은 서버의 loopback 조건에서는 이 항목들이 200~300 ms 수준의 추론보다 작아 전송 빈도·이미지 해상도를 바꾸는 실험은 이번에 추가하지 않았다.

서버 arrival→infer 약 30 ms는 최신 slot에 남은 **선택된 관측의 나이**다. 계속 관측을 교체하는 로봇이 서비스를 기다린 전체 시간과 다르다. payload 크기를 근거로 실제 네트워크에서도 문제가 없다고 일반화하면 안 된다.

**Nsight 수집 방법과 읽을 때의 제한**

Nsight Systems 2025.6.3, CUDA/NVTX만 기록했다. CPU sampling·context switch 수집은 끄고 시스템 perf 설정은 바꾸지 않았다. warmup과 첫 실제 요청 이후 5초를 기다려 약 15초를 수집했다. 각 profile의 client rollout은 60초다. 추론·input 준비·JIT sample 호출·NumPy materialize·output transform에 NVTX 구간을 넣고 서버 batch ID, episode/request ID, broker queue 이벤트와 연결했다. 프로파일러 실행의 처리량은 위 180초 기준 비교에 포함하지 않는다.

원래 graph 모드 B2 trace는 정상 변환됐다. B4 raw trace에는 `CudaDeviceGraph`의 종료 timestamp가 시작보다 작은 이벤트가 있어 importer가 실패했다. 설치되어 있던 2026.1 importer도 같은 raw를 복구하지 못했다. 원본 QDSTRM과 실패 로그, 유효하지 않은 작은 report는 보존하고 데이터의 timestamp를 수정하지 않았다. `nsys stop`이 exit 0이어도 import 실패가 있을 수 있어 report 존재·크기와 오류 로그를 확인하도록 runner를 보강했다.

그 뒤 B2/B4 모두 **node 모드**로 같은 조건에서 재수집했다. 첫 node 시도는 `nsys start`에 application 옵션을 중복 전달해 수집 시작 전에 실패했고 해당 로그도 별도 폴더에 보존했다. 옵션은 최초 `profile` 명령에 두도록 수정했다. [NVIDIA 문서](https://docs.nvidia.com/nsight-systems/UserGuide/)에 따라 node 기록은 graph 내부 활동을 볼 수 있지만 측정 부담이 더 클 수 있다. 기존 graph B2와 node B4를 동일 조건의 성능 비교로 섞지 않는다.

`sample_dispatch`라는 NVTX 이름은 JIT sample 함수 호출 전체를 감싼다. graph B2의 실제 B2에서는 평균 197.52 ms 중 같은 host thread의 `cuStreamSynchronize`가 166.10 ms였고, 그중 기록된 CUDA 활동과 163.03 ms가 겹쳤다. 즉 이 구간을 순수 CPU enqueue 시간으로 해석하거나 동기화 시간을 GPU 시간에 더하면 안 된다. input 준비에는 변환·noise·device batch 준비가 함께 포함되어 있으며 CPU sampling 없이 세부 CPU 원인을 확정할 수 없다.

CUDA interval은 겹치는 구간을 합집합으로 계산한다. graph 모드의 graph span은 내부 공백을 포함할 수 있고, node 모드도 추적한 서버 worker의 활동만 나타낸다. 별도 LIBERO 프로세스의 렌더링은 이 trace에 포함되지 않는다. 어떤 모드에서도 이 값을 GPU 전체 utilization 또는 SM occupancy라고 부르지 않는다.

**유효한 node trace의 비교 결과**

| 지표 | 최대 batch 2 | 최대 batch 4 |
| --- | ---: | ---: |
| 수집 길이, s | 15.356 | 15.341 |
| 완전히 기록된 infer_batch | 66 | 65 |
| 완전한 추론의 첫 시작~마지막 종료 중 infer 비율 | 97.74% | 97.69% |
| 추론 사이 공백 평균, ms | 5.25 | 5.39 |
| 부족 시작 횟수 | 104 | 91 |
| 회복 chunk 추론 중 시작 | 100 | 86 |
| 회복 chunk 추론 전 시작 | 4 | 2 |
| 회복 chunk 추론 후 시작 | 0 | 3 |

capture 경계에 걸린 불완전 NVTX range는 제외했다. Nsight와 서버 기록의 시작·끝 시각은 모두 1 ms 이내로 일치했다. node B2/B4의 같은 actual B2가 각각 231.08/224.46 ms인 차이는 서로 다른 실행·입력·열 상태를 포함한다. cap 변경이 동일 입력의 B2 kernel을 빨라지게 했다는 인과 해석은 할 수 없다. 부족 시작 횟수는 짧은 profile 구간의 진단값이며 180초 기준 결과의 우열을 뒤집는 근거가 아니다.

| 최대 batch | 실제 B | 완전한 추론 수 | NVTX 평균 ms | NVTX p95 ms |
| --- | ---: | ---: | ---: | ---: |
| 2 | 1 | 7 | 157.22 | 172.88 |
| 2 | 2 | 59 | 231.08 | 259.25 |
| 4 | 1 | 15 | 158.30 | 168.22 |
| 4 | 2 | 35 | 224.46 | 241.73 |
| 4 | 3 | 12 | 284.16 | 301.90 |
| 4 | 4 | 3 | 326.67 | 328.55 |

max B4에서도 B4 자체는 3회만 포착됐다. 이 세 호출의 p95는 표본 수가 작으므로 안정적인 tail latency 추정으로 사용할 수 없다.

| 최대 batch / 실제 B | 준비 ms | sample 함수 호출 ms | materialize ms | 출력 변환 ms |
| --- | ---: | ---: | ---: | ---: |
| 2 / 1 | 27.65 | 127.81 | 1.36 | 0.16 |
| 2 / 2 | 28.68 | 199.89 | 2.11 | 0.18 |
| 4 / 1 | 28.90 | 127.75 | 1.33 | 0.12 |
| 4 / 2 | 25.04 | 197.34 | 1.72 | 0.16 |
| 4 / 3 | 29.41 | 252.48 | 1.81 | 0.24 |
| 4 / 4 | 23.80 | 301.92 | 0.60 | 0.18 |

계측한 구간에서는 sample 함수 호출이 가장 길었고 그 안의 CUDA 실행과 host 동기화가 겹쳤다. materialize에서 대부분 시간을 기다린다는 단순한 설명과는 맞지 않는다. 각 batch별 host sync와 CUDA 합집합·겹침은 `adapter_phases.csv`에 따로 저장했다.

**결정 시점의 예측과 실제 결과**

scheduler가 결정을 내릴 때의 inference duration, completion, arrival, 예상 chunk 시작/skip을 값으로 복사해 기록했다. 이후 latency tracker가 갱신된 값으로 과거 예측을 재계산하지 않았다. capture 안의 완전한 batch만 대상으로 하고 예상·실제 batch 크기가 같은 경우를 비교했다(크기 불일치 0). 추론 오차는 요청 수로 중복 가중하지 않도록 batch ID로 중복 제거했다. 아래 양수는 실제 결과가 예측보다 늦음을 뜻한다.

| 최대 batch / 실제 B | 추론 수 | 추론 오차 평균 ms | 추론 오차 p95 ms | 수용 응답의 도착 오차 평균 ms |
| --- | ---: | ---: | ---: | ---: |
| 2 / 1 | 7 | 9.00 | 27.99 | 11.09 |
| 2 / 2 | 59 | 1.64 | 26.72 | 2.96 |
| 4 / 1 | 15 | 8.06 | 17.03 | 11.00 |
| 4 / 2 | 35 | 4.45 | 17.89 | 6.57 |
| 4 / 3 | 12 | 14.32 | 28.73 | 17.83 |
| 4 / 4 | 3 | 20.16 | 20.60 | 21.31 |

이 짧은 profile 구간에서는 평균 추론·도착을 다소 빠르게 예측했다. 하지만 profiler 자체와 온도 변화가 latency tracker에 영향을 줄 수 있고 큰 B의 표본이 적으므로, 이 수치만으로 일반 실행의 부족을 예측기 편향 탓으로 확정하거나 latency margin을 변경하지 않았다.

수용 응답의 실제 chunk 시작 action index는 예측보다 1 큰 경우가 B2 조건 113/123개, B4 조건 116/130개였다. 같은 index는 9/11개, 2 큰 경우는 1/3개다. 따라서 예측 `H - first_executed_index`와 실제 수용 가능한 suffix의 개수 차이를 도착 지연 오차만으로 해석할 수 없다. **수용 가능한 suffix 개수, 기존 queue를 뺀 순증가량, 최종적으로 실행한 개수는 서로 다른 지표**다. `response_predictions.csv`에 실제 시작 index 차이와 실제 queue 순증가량을 함께 저장했다. 다음 예측 모델 개선에서는 mirror가 예상한 관측/action 기준점과 실제 slot 선택의 기준점을 먼저 대조해야 한다. 현재 기록은 기준점이 다르다는 관측이며 원인을 off-by-one 버그로 확정한 것은 아니다.

도표의 확대 구간은 capture 양쪽 1초 여유가 있는 부족 시작 중 가장 긴 구간, 동률이면 가장 이른 구간을 자동 선택했다. B2에서는 robot 1의 6 step(300.91 ms) 부족이 회복 추론 시작보다 32.76 ms 먼저 시작했고, 추론 종료 뒤 약 1.97 ms에 응답을 수용했다. B4에서는 robot 1의 5 step(250.14 ms) 부족이 회복 추론보다 2.32 ms 먼저 시작했고 종료 뒤 약 1.97 ms에 수용했다. 수용 직후 즉시 새 step을 실행하는 것이 아니라 다음 control tick까지 기다리므로 붉은 구간이 수신 marker 뒤까지 이어질 수 있다. queue가 마지막 action pop으로 0이 된 시각과 첫 null action 시각도 한 tick 다를 수 있다.

이 결과는 서버가 추론을 거의 연속 수행하는 동안 각 로봇 queue가 고갈되고, 이후 chunk가 일부 prefix를 건너뛴 채 queue를 보충하는 경로를 보여준다. 다음 개선을 검증할 때에는 현재 B2 기준선을 유지하고, 예측의 기준점 정렬 및 유효 action 보충량을 바꾼 조건을 하나씩 비교하는 것이 타당하다. 이번에는 원인 분리를 위한 실험을 완료했으며 scheduler 목적 함수나 horizon을 변경하는 새 정책 실험은 포함하지 않았다.

node B2의 60초 rollout 종료부에 처리됐으나 구조화된 send/discard/ACK 기록이 없는 응답 1개(request 4700)가 있다. 추론은 capture 종료 38초 이상 뒤이며 해당 episode의 마지막 기록 step 이후다. 같은 종료부 stdout에는 연결이 없어 응답을 버렸다는 로그가 있지만 request ID가 없으므로 구조화된 경로 집계에서는 미확인 1개로 유지했다. node B4는 0개이며 두 capture 내부에는 미확인 응답이 없다. 서버 정리 시 SIGINT에 따른 KeyboardInterrupt/CancelledError 로그도 보존했다. 이를 추론 중 실패나 일반 실행의 전송 손실로 집계하지 않았다.

**재현과 원본 위치**

기준 실행은 `output/local_followup_20260915/run_r4_b{2,4}_rep{1,2,3}`에, 고정 입력·측정은 각각 `output/fixed_inputs_20260915`, `output/fixed_batch_20260915`에 있다. 유효한 node trace는 `output/local_followup_nodes_20260915/profile_r4_b{2,4}_rep0`에 보관한다. 각 manifest에 전체 명령·GPU·commit·시각을 남겼다. 이 폴더들은 Git에 올리지 않는다.

새로 재현할 때는 아래 output 이름을 아직 없는 폴더로 바꾸고 사용 가능한 GPU를 지정한다. runner의 GPU guard는 compute 프로세스 확인이며 연구실 예약 시스템을 대신하지 않는다.

```bash
.venv/bin/python -m scripts.local_batch_sweep --phase clean \
  --output output/repro_baseline --gpu 0 --seconds 180 --repeats 3
.venv/bin/python -m scripts.benchmark_fixed_batch capture \
  --output output/repro_inputs --gpu 0
.venv/bin/python -m scripts.benchmark_fixed_batch run \
  --inputs output/repro_inputs --output output/repro_fixed --gpu 0 \
  --samples 50 --repeats 3 --with-renderer-control
.venv/bin/python -m scripts.local_batch_sweep --phase profile4 \
  --output output/repro_nsight --gpu 0 --graph-trace node
```

분석 도구는 `analyze_local_batch_sweep`, `analyze_chunk_usage`, `analyze_followup_events`, `analyze_fixed_batch`, `analyze_nsight_followup`이다. Nsight는 먼저 `nsys export --type sqlite --output <run>/timeline.sqlite <run>/timeline.nsys-rep`로 변환하고, `<run>`에 대해 `analyze_followup_events` 다음 `analyze_nsight_followup`을 실행한다. 주요 도표는 기준 root의 `comparison.png`, `chunk_usage/chunk_use_comparison.png`, fixed root의 `batch_cost.png`와 `renderer_control.png`, 각 trace의 `nsight_analysis/capture.png`, `starvation_detail.png`, `adapter_phases.png`이며 PDF도 함께 저장한다.

실험 중 지속 GPU telemetry에서 software thermal slowdown이 간헐적으로 Active였고 clock이 변했다. 해당 telemetry는 기준 실행 중간부터 시작했으므로 모든 실행 전 구간을 보장하지 않는다. [NVIDIA의 clock event 정의](https://docs.nvidia.com/deploy/nvidia-smi/index.html)에 따라 이를 제한 요인으로 기록한다. 고정 clock 조건의 하드웨어 최대 성능 결과는 아니다. 실행 순서 교대, 세 block 및 off/on/off가 변동 영향을 줄이지만 제거하지는 못한다. 측정 부담·동일 seed 반복·공유 GPU의 시뮬레이터 부하를 함께 고려해야 한다.

검증: serving·episode race·scheduler broker·OpenPI adapter 관련 37개 테스트 통과. 코드와 재현 도구·보고서는 개인 fork의 `experiment/mock-communication` 브랜치에 커밋하고 push했다. 가중치·원시 trace·개인 캐시는 올리지 않았다.
