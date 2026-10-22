# Evolução de Performance e Otimizações Arquiteturais (`IMPROVEMENTS_PROGRESS.md`)

**Projeto:** Decomposição Paralela de Birkhoff-von Neumann (BvN) via Coloração de Grafos Eulerianos  
**Disciplina:** Programação e Arquitetura de Computadores Paralelos — DCC/UFMG  
**Professor:** George Luiz Medeiros Teodoro  
**Aluno:** João Vitor Soares (`2023002138`)  
**Hardware de Desenvolvimento/Teste Local:** AMD Ryzen 5 5500U (6 núcleos físicos / 12 threads SMT, Cache L1d 32 KB/core, L2 512 KB/core, L3 8 MB)  
**Verificação Matemática:** `PASSED (100% Exact)` em todos os commits, modos e padrões de tráfego.

---

## 1. Linha do Tempo de Commits e Otimizações Implementadas

| Ordem | Commit | Escopo | Técnica de Otimização Implementada | Problema Arquitetural Resolvido |
| :---: | :---: | :--- | :--- | :--- |
| **0** | `1b45ea8` / `5fb939a` | Baseline Inicial | Refatoração modular (`types`, `traffic`, `regularizer`, `euler_trail`, `decomposer_*`, `postprocessor`) + `alignas(64)` + Parity Pre-Halving + Flat CSR. | Eliminação de *False Sharing* nas linhas de cache de 64B e redução da multiplicidade par antes do ciclo de Hierholzer. |
| **1** | `73b76a0` | Memória / TLS | **`thread_local EulerWorkspace` (`get_thread_workspace()`)** com pré-alocação por profundidade (`level_red[depth]`, `level_blue[depth]`) e *warm-up* fora do cronômetro. | Eliminação de milhares de chamadas concorrentes ao alocador `malloc`/`free` (glibc arena locks) e *page faults* do SO nas subárvores recursivas. |
| **2** | `fbc51ce` | Algorítmica / Poda | **Poda de Simetria Exata (*Symmetry Pruning*, `E_odd == 0`)**: quando todas as arestas têm peso par, o subgrafo ímpar é vazio e $M_{\text{red}} \equiv M_{\text{blue}}$. | Corta metade da árvore de recursão em matrizes com pesos pares/simétricos (`skewed` e `bursty`), copiando as folhas prontas em $O(N \cdot D/2)$. |
| **3** | `38e9531` | Estrutura de Dados / ISA | **Substituição do Flat CSR por Bitmask 64-bit (`adj_L`, `adj_R`, `red_mask`) + Instruções `TZCNT` (`__builtin_ctzll`) e `BLSR` (`mask & (mask - 1)`)** + Caminhamento Euleriano direto de 2 passos ($L \to R \to L$) e *Self-Zeroing*. | Elimina *pointer/index chasing*, *branch mispredictions* e o vetor `edge_state`. O próprio ciclo Euleriano zera 100% dos bits sem `memset`. |
| **4** | `cbcd121` | Paralelismo Nível 0 | **Split Paralelo Lock-Free na Raiz (`euler_root_split_omp_arch`)**: construção paralela do Bitmask em blocos de 64 linhas (`schedule(static, 64)`) + extração paralela de `out_red`/`out_blue` via *prefix-sum*. | Como `u >> 6` é disjunto a cada bloco de 64 linhas, as 12 threads gravam em words `uint64_t` disjuntas de `adj_R` sem `atomic` e sem *locks*. |
| **5** | `fb827df` | Folhas, Layout 8B e Bitmask 2 Níveis | 1. **Fast-Path `degree == 2` (`extract_degree2_leaves`)**.<br>2. **`WeightedEdge` compactado em 8 Bytes (`alignas(8)`)**.<br>3. **Bitmask Hierárquico de 2 Níveis (`summary_L`/`summary_R` com duplo `TZCNT`)**.<br>4. **Pool `node_buffers` nos nós internos OpenMP**. | Corta os últimos $D/2$ splits da árvore de recursão; aumenta densidade para 8 arestas por cache line (64B); elimina varredura linear de words vazias. |
| **6** | *Atual* | Emissão Direta no Walk + Novo Modo `omp_block` | 1. **Single-Pass Walk Emission**: emite arestas pares no 1º scan e arestas ímpares unitárias (`w == 1`) direto durante o caminhamento Euleriano (sem 2º scan de `edges`).<br>2. **Novo Modo `--mode omp_block` com Seletor Dinâmico $K \times K$ (`select_optimal_block_factor`)**: fatia a matriz $N \times N$ em $K^2$ sub-matrizes $(N/K) \times (N/K)$ residentes em Cache L1d (32 KB), aplica *Phantom Padding* local em paralelo e funde as diagonais disjuntas (*Latin-Square Block Decomposition*). | Elimina o gargalo de 1, 2 e 4 threads nos níveis superiores da árvore ($D \to D/2 \to D/4$) e mantém cada sub-grafo ($256 \times 256$, `8 KB`) 100% dentro da Cache L1d de cada núcleo. |

---

## 2. Evolução dos Tempos em $N = 256, D = 256$ (12 Threads)

 Comparação da Fase 2 & 3 (Euler + BvN) desde o baseline até a versão atual:

| Padrão | `seq` (Original) | `omp_naive` (12t) | `omp_arch` (Original) | `omp_arch` (Atual, 12t) | Speedup Total (`omp_arch` vs `seq` orig.) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`uniform`** | `14.12 ms` | `14.08 ms` | `6.87 ms` | **`2.19 ms`** | **6.45x** |
| **`skewed`** | `9.29 ms` | `12.66 ms` | `4.19 ms` | **`0.46 ms`** | **20.19x** |
| **`bursty`** | `10.97 ms` | `15.72 ms` | `4.91 ms` | **`0.91 ms`** | **12.05x** |

---

## 3. Evolução dos Tempos em $N = 1024, D = 1024$ (12 Threads)

### 3.1. Tempo da Fase 2 & 3 (Euler + BvN) em Milissegundos (`ms`)

| Etapa / Versão | `uniform` (Fase 2&3) | `skewed` (Fase 2&3) | `bursty` (Fase 2&3) |
| :--- | :---: | :---: | :---: |
| **1. Baseline Original (`seq`, 1t)** | `292.20 ms` | `140.42 ms` | `173.49 ms` |
| **2. OpenMP Ingênuo (`omp_naive`, 12t)** | `138.33 ms` | `144.61 ms` | `130.82 ms` |
| **3. `omp_arch` Inicial (CSR + Cutoff, 12t)** | `164.08 ms` | `78.50 ms` | `84.12 ms` |
| **4. + Bitmask 64-bit + Root Split (`cbcd121`, 12t)** | `68.56 ms` | `29.10 ms` | `46.52 ms` |
| **5. + Fast-Path `deg=2` + 8B Edge + 2-Level Bitmask (`fb827df`, 12t)** | `51.14 ms` | `14.67 ms` | `27.36 ms` |
| **6. + Single-Pass Walk (`seq` Otimizado, 1t)** | `130.81 ms` | `42.44 ms` | `65.90 ms` |
| **7. + Single-Pass Walk (`omp_arch`, 12t)** | `43.98 ms` | **`11.75 ms`** | **`13.00 ms`** |
| **8. + Novo Modo `omp_block` ($K \times K$ Dinâmico em L1, 12t)** | **`15.32 ms`** | `19.28 ms` (`14.81 ms` c/ $K=16$) | `42.94 ms` |

### 3.2. Resumo dos Ganhos Máximos Atingidos em $N = 1024, D = 1024$
- **`uniform`:** Fase 2 & 3 caiu de `292.20 ms` (`seq` original) para **`15.32 ms`** (`omp_block`, $K=4$, 16 blocos de $256 \times 256$) $\rightarrow$ **Speedup de 19.07x** (e **Tempo Total de Pipeline de `19.30 ms`**).
- **`skewed`:** Fase 2 & 3 caiu de `140.42 ms` (`seq` original) para **`11.75 ms`** (`omp_arch`) e **`14.81 ms`** (`omp_block`) $\rightarrow$ **Speedup de 11.95x**.
- **`bursty`:** Fase 2 & 3 caiu de `173.49 ms` (`seq` original) para **`13.00 ms`** (`omp_arch`) $\rightarrow$ **Speedup de 13.34x**.

---

## 4. Análise Arquitetural: `omp_arch` vs `omp_block`

1. **Por que o `omp_block` domina no tráfego `uniform` (`15.32 ms` vs `43.98 ms`)?**
   - Na decomposição global (`omp_arch`), os 3 primeiros níveis da árvore ($D=1024 \to 512 \to 256$) possuem apenas 1, 2 e 4 subproblemas, subutilizando as 12 threads justamente quando os grafos são mais densos, além de exigir um bitmask de $1024 \times 16 \times 8\text{B} = 128\text{ KB}$ (que transborda da Cache L1d de 32 KB para a L2).
   - No `omp_block` com $K=4$ (selecionado dinamicamente por `select_optimal_block_factor`), a matriz $1024 \times 1024$ é particionada em **16 sub-matrizes independentes de $256 \times 256$** agrupadas em 4 rodadas diagonais disjuntas (*Latin-Square*).
   - Cada bloco $256 \times 256$ requer apenas $256 \times 4 \times 8\text{B} = \mathbf{8\text{ KB}}$ para `adj_L` e `8 KB` para `adj_R` (**16 KB total, 100% residente na Cache L1d de 32 KB**), e todas as 12 threads já iniciam no instante $t=0$ com 16 tarefas independentes sem qualquer contenção.
2. **Trade-off em Tráfegos com Hotspots (`bursty` / `skewed`):**
   - Quando uma porta concentra quase toda a sua demanda num único quadrante $(N/K) \times (N/K)$, a soma máxima local daquele quadrante excede $D/K$. Regularizar cada quadrante separadamente para a próxima potência de 2 infla a soma dos graus das $K$ rodadas diagonais ($D_{\text{total}} > D$).
   - Por isso, o `omp_arch` global (que regulariza a matriz inteira uma única vez para $D=1024$ e explora *Symmetry Pruning*) atinge o menor tempo em `skewed` (`11.75 ms`) e `bursty` (`13.00 ms`), enquanto o `omp_block` atinge o menor tempo absoluto em `uniform` (`15.32 ms`).
