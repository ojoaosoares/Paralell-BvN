# Parallel-BvN: Decomposição de Birkhoff-von Neumann via Coloração Euleriana Paralela

**Disciplina:** Programação e Arquitetura de Computadores Paralelos (DCC/UFMG)  
**Professor:** George Luiz Medeiros Teodoro  
**Aluno:** João Vitor Soares (Matrícula: 2023002138)  

Proposta completa disponível em: [`docs/PROPOSTA.md`](docs/PROPOSTA.md).

---

## Visão Geral da Arquitetura do Pipeline

1. **Fase 1 (`src/phase1_padding.cpp`):** Regularização (*Padding*) com redução paralela/SIMD de somas de linhas e colunas em $O(N^2)$ e inserção de tráfego fantasma em $O(N)$, gerando um multigrafo bipartido $D$-regular ($D = 2^k$).
2. **Fases 2 e 3 (`src/phase2_3_euler_bvn.cpp`):** Divisão e Conquista Euleriana ($D \to D/2 \to \dots \to 1$) e extração das folhas de Grau 1 (matrizes de permutação). Possui três modos para comparação microarquitetural:
   * `seq`: Baseline sequencial em *Flat CSR*.
   * `omp_naive`: OpenMP ingênuo (arestas unitárias expandidas, cursores contíguos sofrendo *false sharing* em linhas de cache e criação irrestrita de `#pragma omp task` até $D=1$).
   * `omp_arch`: OpenMP orientado à arquitetura:
     * Cursores atômicos isolados por linha de cache (`alignas(64)`) eliminando *false sharing* entre núcleos/sockets.
     * Pré-divisão de multiplicidades pares ($w = 2q + r$) reduzindo drasticamente o volume de operações atômicas.
     * Costura determinística *lock-free* de caminhadas abertas no super-grafo Euleriano.
     * *Task Cut-off* configurável (`--cutoff`) mantendo subgrafos pequenos nos caches L1/L2 locais.
3. **Fase 4 (`src/phase4_cleanup.cpp`):** Filtragem paralela *lock-free* do tráfego fantasma (substituição por `-1` / *no-op*) e verificador matemático exato (`verify_decomposition`).

---

## Compilação e Execução Local

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Executar modo orientado à arquitetura (omp_arch)
./build/bvn_bench --mode omp_arch --n 256 --d 256 --threads 8 --cutoff 8

# Comparar com modo ingênuo (omp_naive) e sequencial (seq)
./build/bvn_bench --mode omp_naive --n 256 --d 256 --threads 8
./build/bvn_bench --mode seq --n 256 --d 256
```

## Execução Automatizada no CloudLab (Multi-Socket NUMA + `perf`)

Em um nó bare-metal do CloudLab (ex.: Dual-Socket AMD EPYC `c6525-100g`, `r6525` ou Intel Xeon `c6420`):

```bash
chmod +x scripts/run_cloudlab_bench.sh
./scripts/run_cloudlab_bench.sh
```
