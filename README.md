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
│   ├── decomposer.hpp            # Interfaces de decomposição BvN (seq, omp_naive, omp_arch, omp_block)
│   ├── euler_parallel.hpp        # Partição Euleriana paralela no topo da árvore (--par-walk-min)
│   └── postprocessor.hpp         # Filtragem de tráfego fantasma e verificador matemático exato
├── src/
│   ├── main.cpp                  # CLI de benchmark e relatório de métricas
│   ├── traffic.cpp               # Implementação do gerador de matrizes de demanda
│   ├── regularizer.cpp           # Fase 1: Regularização de grau D = 2^k
│   ├── euler_trail.cpp           # Reconciliação de caminhadas abertas no super-grafo Euleriano
│   ├── decomposer_seq.cpp        # Fases 2 e 3: Baseline Sequencial em Flat CSR
│   ├── decomposer_naive.cpp      # Fases 2 e 3: OpenMP Ingênuo (expondo false sharing e task overhead)
│   ├── decomposer_arch.cpp       # Fases 2 e 3: OpenMP Orientado à Arquitetura (alignas(64), cut-off, NUMA) e omp_block
│   ├── euler_parallel.cpp        # Coloração Euleriana paralela com CAS + union-find com paridade
│   └── postprocessor.cpp         # Fase 4: Limpeza lock-free de tráfego fantasma e validação
├── profile.py                    # Perfil do CloudLab (geni-lib): nó bare-metal c6420 + setup automático
└── scripts/
    ├── cloudlab/setup.sh         # Setup no boot do nó: dependências, perf, governor e build Release
    ├── cloudlab_run.sh           # Driver único: dependências, smoke test, suíte e tarball (quick | full)
    ├── run_cloudlab_bench.sh     # Suite automatizada de benchmarks e perf stat para nós CloudLab
    ├── run_statistical_bench.py  # Comparação de modos, escala extrema, strong scaling e padrões (30 repetições)
    └── run_advanced_bench.py     # 5 baterias: perf stat, NUMA/afinidade, esparsidade, grid K×K, weak scaling
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

### Perfil do CloudLab a partir deste repositório

O arquivo [`profile.py`](profile.py) define a topologia: um nó bare-metal com Ubuntu 22.04,
por padrão `c6420` (Clemson, Dual-Socket Intel Xeon Gold 6142, 384 GB de RAM), o nó dos
resultados atuais. No boot, o CloudLab clona o repositório em `/local/repository` e executa
[`scripts/cloudlab/setup.sh`](scripts/cloudlab/setup.sh), que instala as dependências
(`cmake`, `numactl`, `hwloc`, `perf`), libera o `perf` para usuários comuns,
fixa o governor da CPU em `performance` e compila o `bvn_bench` em Release.

1. No portal do CloudLab: **Experiments › Create Experiment Profile**.
2. Em **Source**, escolha **Git Repository** e informe
   `https://github.com/ojoaosoares/Paralell-BvN.git`.
3. Instancie o perfil. Parâmetros: tipo de hardware, imagem de disco, benchmark ao final
   do setup (`none`, `quick` ou `full`, via `scripts/cloudlab_run.sh`) e governor `performance`.
4. Acompanhe o setup com `tail -f /local/logs/bvn-setup.log` e o benchmark com
   `tail -f /local/logs/bvn-bench.log`.

O binário fica em `/local/repository/build/bvn_bench`. A topologia detectada
(`lscpu`, `numactl --hardware`) é salva em `/local/logs/`: nós `c6420` podem vir
com 2 ou 4 nós NUMA, o que muda os resultados.

Nós recomendados (bare-metal, multi-socket): `r6525` (2× AMD EPYC 7543, Zen 3, 2 sockets × 8 CCDs) como alvo principal e `c6420` (2× Intel Xeon Gold 6142, Skylake-SP) como contraste Intel. O `c6525-25g`/`c6525-100g` é **single-socket** (EPYC 7302P/7402P) e não exercita NUMA entre sockets. Confira a especificação atual na página de hardware do CloudLab antes de reservar. No perfil, escolha o nó pelo parâmetro **Tipo de hardware**.

Caminho mais curto em um nó novo (instala dependências, libera o `perf`, roda um teste de corretude e empacota os resultados em `bvn_<host>_<modo>_<data>.tar.gz`):

```bash
tmux new -s bvn                     # evita perder a execução se o SSH cair
./scripts/cloudlab_run.sh quick     # ~15-30 min
./scripts/cloudlab_run.sh full      # ~1-3 h
```

Execução manual da suíte:

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

Suítes estatísticas (usadas nos resultados da apresentação; usam o binário já compilado em `build/bvn_bench`):

```bash
python3 scripts/run_statistical_bench.py # 30 repetições por ponto -> results_statistical_*/
python3 scripts/run_advanced_bench.py    # 5 baterias arquiteturais -> results_advanced_*/
```

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

## Divisão Euleriana Paralela no Topo da Árvore (`--par-walk-min`, experimental)

No `omp_arch`, o nível k da árvore tem só 2^k subárvores; enquanto 2^k < T, cada split serial deixa threads ociosas. Com `--par-walk-min N`, splits do topo com ≥ N arestas (e a raiz) trocam a caminhada serial por uma partição Euleriana paralela:

1. Em cada vértice, as arestas ímpares são emparelhadas (pares consecutivos); arestas pareadas precisam de cores diferentes, e a relação forma ciclos pares.
2. Tasks (`omp taskloop`) crescem segmentos desses ciclos em paralelo, reivindicando arestas com CAS e colorindo-as relativamente ao início do segmento.
3. Um union-find com paridade (sequencial, sobre os segmentos) decide qual segmento inverte as cores; o resultado é aplicado em paralelo.

Fica **desligado por padrão**. Em um Ryzen 5 5500U (6 cores), N=2048, a coloração paralela da raiz leva ~27 ms contra ~12 ms da caminhada serial em bitmask: a caminhada usa 1 bit por par (u, v) residente em L2/L3, enquanto a partição precisa de ~16 B por aresta ímpar. O script mede a ablação (`par_walk_ablation.csv`) para verificar se, com dezenas de cores, o paralelismo extra compensa.
