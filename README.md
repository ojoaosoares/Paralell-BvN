# Parallel-BvN: Decomposição de Birkhoff-von Neumann via Coloração Euleriana Paralela

**Disciplina:** Programação e Arquitetura de Computadores Paralelos (DCC/UFMG)  
**Professor:** George Luiz Medeiros Teodoro  
**Aluno:** João Vitor Soares (Matrícula: 2023002138)  

Proposta completa disponível em: [`docs/PROPOSTA.md`](docs/PROPOSTA.md).

---

## Organização do Repositório

```text
├── CMakeLists.txt
├── docs/
│   └── PROPOSTA.md               # Proposta do Trabalho Prático (Foco em Arquitetura NUMA/Cache)
├── include/bvn/
│   ├── types.hpp                 # Estruturas base (AlignedCursor alignas(64), FlatEdge, Config)
│   ├── traffic.hpp               # Gerador de matrizes de tráfego (uniform, skewed, bursty)
│   ├── regularizer.hpp           # Regularização / Padding O(N^2) SIMD + O(N) two-pointer
│   ├── euler_trail.hpp           # Costura determinística lock-free de caminhadas abertas
│   ├── decomposer.hpp            # Interfaces de decomposição BvN (seq, omp_naive, omp_arch)
│   └── postprocessor.hpp         # Filtragem de tráfego fantasma e verificador matemático exato
├── src/
│   ├── main.cpp                  # CLI de benchmark e relatório de métricas
│   ├── traffic.cpp               # Implementação do gerador de matrizes de demanda
│   ├── regularizer.cpp           # Fase 1: Regularização de grau D = 2^k
│   ├── euler_trail.cpp           # Reconciliação de caminhadas abertas no super-grafo Euleriano
│   ├── decomposer_seq.cpp        # Fases 2 e 3: Baseline Sequencial em Flat CSR
│   ├── decomposer_naive.cpp      # Fases 2 e 3: OpenMP Ingênuo (expondo false sharing e task overhead)
│   ├── decomposer_arch.cpp       # Fases 2 e 3: OpenMP Orientado à Arquitetura (alignas(64), cut-off, NUMA)
│   └── postprocessor.cpp         # Fase 4: Limpeza lock-free de tráfego fantasma e validação
└── scripts/
    └── run_cloudlab_bench.sh     # Suite automatizada de benchmarks e perf stat para nós CloudLab
```

---

## Compilação e Execução Local

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Executar modo orientado à arquitetura (omp_arch)
./build/bvn_bench --mode omp_arch --n 512 --d 512 --threads 6 --cutoff 8

# Comparar com modo ingênuo (omp_naive) e sequencial (seq)
./build/bvn_bench --mode omp_naive --n 512 --d 512 --threads 6
./build/bvn_bench --mode seq --n 512 --d 512
```

## Execução Automatizada no CloudLab (Multi-Socket NUMA + `perf`)

Nós recomendados (bare-metal, multi-socket): `r6525` (2× AMD EPYC 7543, Zen 3, 2 sockets × 8 CCDs) como alvo principal e `c6420` (2× Intel Xeon Gold 6142, Skylake-SP) como contraste Intel. O `c6525-25g`/`c6525-100g` é **single-socket** (EPYC 7302P/7402P) e não exercita NUMA entre sockets. Confira a especificação atual na página de hardware do CloudLab antes de reservar.

```bash
sudo sysctl kernel.perf_event_paranoid=-1   # libera contadores de hardware para o perf
./scripts/run_cloudlab_bench.sh

# Variáveis opcionais
REPS=20 N_LIST="1024 2048" PATTERNS="uniform bursty" ./scripts/run_cloudlab_bench.sh
```

O script gera em `results_<timestamp>/`:

| Arquivo | Conteúdo |
|---|---|
| `scaling_results.csv` | Escalabilidade forte (`seq`, `omp_naive`, `omp_arch`, `omp_block`; binding `spread`/`close`) |
| `domain_mapping.csv` | Mapeamento NUMA/L3: `spread` vs `numa` vs `l3` vs `interleave` (`numactl --interleave=all`) |
| `cutoff_sensitivity.csv` | Varredura do *task cut-off* do `omp_arch` |
| `perf_<modo>_n<N>.txt` | `perf stat -r REPS` com os eventos suportados pela CPU |
| `places_numa.txt`, `places_l3.txt` | Listas explícitas de *places* geradas a partir do `lscpu` |

Colunas do CSV: `rep,binding,mode,pattern,N,D,threads,cutoff,phase1_ms,phase2_3_ms,phase4_ms,total_ms,schedules,frame_eff,verified`.
`schedules` é o tamanho do frame (nº de permutações) e `frame_eff = pacotes reais / (N × schedules)`. O `omp_block` pode inflar o frame (padding por bloco até potência de 2), então compare tempo **e** `frame_eff`. No `omp_block` a limpeza de tráfego fantasma é feita dentro da fase 2/3, então use `total_ms` para comparar modos.

## Mapeamento em Domínios NUMA / L3 (`--domains`)

`--domains G` divide as threads em G grupos; cada grupo é um *team* aninhado (`proc_bind(spread)` externo, `proc_bind(close)` interno) preso a uma partição da lista de *places*. `--domains 0` cria um grupo por *place*.

- `omp_arch`: as duas subárvores independentes do nível 1 (Red/Blue) rodam cada uma em um socket (G ≤ 2).
- `omp_block`: os *shifts* da diagonal em Quadrado Latino são divididos em G faixas contíguas; extração, decomposição, limpeza e fusão de um *shift* ficam no mesmo domínio.

```bash
# Um grupo por NUMA node (um CPU por core físico em cada place)
OMP_PLACES="{0,2,4,...},{64,66,...}" OMP_PROC_BIND=spread \
    ./build/bvn_bench --mode omp_block --n 2048 --d 2048 --threads 64 --domains 0
```

Use listas explícitas de *places* (o script as gera): em algumas versões do libgomp `OMP_PLACES=ll_caches`/`numa_domains` colapsa tudo em um único *place*, prendendo todas as threads ao primeiro CCX.
