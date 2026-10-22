# Proposta de Trabalho Prático
**Disciplina:** Programação e Arquitetura de Computadores Paralelos  
**Professor:** George Luiz Medeiros Teodoro  
**Aluno:** João Vitor Soares  
**Matrícula:** 2023002138  

---

## Título
**Paralelização da Decomposição de Birkhoff-von Neumann via Coloração de Grafos Eulerianos com Otimização para Arquiteturas Multi-Socket NUMA**

---

## 1. Descrição do Problema
Em arquiteturas modernas de redes de datacenters e switches ópticos, o escalonamento de pacotes ocorre no modelo *Input-Buffered* (filas na entrada) para evitar gargalos de largura de banda de memória. O grande desafio nesse modelo é resolver colisões de roteamento em tempo real, garantindo o máximo de vazão (*throughput*) sem bloqueios de *Head-of-Line* (HoL).

A decomposição de Birkhoff-von Neumann (BvN) é matematicamente ideal para esse escalonamento, pois garante 100% de *throughput* dividindo matrizes de tráfego duplamente estocásticas (ou sub-estocásticas escaladas) em conjuntos de agendamentos sem conflitos (matrizes de permutação). Contudo, a busca pela decomposição mínima é um problema NP-Completo, e o algoritmo clássico baseado em emparelhamentos bipartidos sucessivos (Hopcroft-Karp / Dulmage-Mendelsohn) possui complexidade da ordem de $O(N^{4.5})$ no pior caso, sendo inerentemente sequencial na extração iterativa de permutações. Devido a essa latência computacional, o BvN clássico torna-se inviável para processamento em tempo real (escala de microssegundos a poucos milissegundos).

O problema a ser implementado consiste em projetar, desenvolver e avaliar microarquiteturalmente uma versão paralela de alto desempenho do algoritmo BvN em **C++ (utilizando OpenMP)**, substituindo a clássica busca sequencial por emparelhamentos bipartidos por um modelo de **Divisão e Conquista através de Particionamento de Ciclos Eulerianos** em multigrafos bipartidos regulares.

O pipeline proposto divide-se em quatro fases:
1. **Regularização (*Padding*):** Etapa rápida $O(N^2)$ que calcula os déficits de linhas e colunas e adiciona tráfego "fantasma" (*dummy traffic*) às demandas reais, transformando uma matriz de tráfego arbitrária em um Multigrafo Bipartido Regular de Grau $D = 2^k$.
2. **Divisão e Conquista Euleriana (Núcleo Paralelo):** Múltiplas threads percorrem caminhadas Eulerianas concorrentes no grafo e colorem as arestas alternadamente (vermelho para transições $L \to R$ e azul para $R \to L$). O grafo de grau $D$ divide-se exatamente em dois subgrafos independentes de grau $D/2$. O processo repete-se recursivamente instanciando novas tarefas (`#pragma omp task`) que mapeiam as subárvores de recursão aos domínios físicos de memória/cache do processador.
3. **Decomposição BvN (*Output*):** As folhas da árvore de recursão atingem grafos bipartidos regulares de Grau 1, que correspondem diretamente a matrizes de permutação exatas sem conflitos.
4. **Limpeza e Pós-processamento:** O tráfego fantasma inserido na Fase 1 é filtrado em paralelo, substituindo conexões artificiais por estados ociosos (*no-op*) e consolidando o cronograma final de acionamento das portas do switch.

---

## 2. Relevância da Solução
A literatura aborda o gargalo do escalonamento de diversas formas, mas geralmente sacrificando o formato do tráfego original ou a exatidão do agendamento:
* Para lidar com rajadas repentinas de tráfego (*bursty traffic*), sistemas como o *Deflection-Compensated BvN* [3] são forçados a realizar desvios físicos complexos de pacotes devido à lentidão do recálculo da matriz.
* Abordagens de dois estágios exigem espalhar e descaracterizar o tráfego original (*Load Balanced BvN Switches* [4]), introduzindo reordenação de pacotes.
* Heurísticas aproximadas de mercado como o *iSLIP* [5] operam rapidamente em hardware, mas perdem eficiência e vazão sob cargas intensas e assimétricas.

Este trabalho ataca a raiz algorítmica e arquitetural do gargalo. Em vez de contornar a lentidão do BvN clássico com heurísticas aproximadas, a proposta explora a redução do problema para coloração 2-regular Euleriana [1, 2] aliada ao paralelismo massivo e à hierarquia de memória de servidores *multicore* modernos (avaliados em nós *bare-metal* do ambiente experimental **CloudLab**). Isso viabiliza o cálculo de decomposições exatas em software para controladores *Software-Defined Networking* (SDN) e escalonadores de circuitos ópticos voltados a cargas de treinamento distribuído de Inteligência Artificial.

---

## 3. Desafios de Paralelização e Foco em Arquitetura de Computadores
Para tornar o trabalho relevante do ponto de vista de **Arquitetura de Computadores Paralelos**, a implementação não se limitará ao uso ingênuo de diretivas OpenMP. Os experimentos serão conduzidos em servidores *bare-metal* multi-socket do **CloudLab** (ex.: nós Dual-Socket AMD EPYC / Intel Xeon com múltiplos domínios NUMA e dezenas de núcleos físicos), atacando quatro problemas fundamentais da interação entre software paralelo e hardware:

### 3.1. Contenção Atômica, Coerência de Cache e Eliminação de *False Sharing*
Na Fase 2, principalmente nos primeiros níveis da árvore de recursão (grau $D$ elevado), múltiplas threads reclamam arestas incidentes a um mesmo conjunto de vértices.
* **Desafio Arquitetural:** Se os cursores de adjacência dos vértices forem armazenados em um vetor contíguo simples (`int cursor[2*N]`), até 16 vértices consecutivos compartilharão a mesma linha de cache de 64 bytes. Operações `#pragma omp atomic capture` executadas por núcleos distintos — especialmente através de sockets diferentes conectados via *AMD Infinity Fabric* ou *Intel UPI* — causarão invalidações massivas de linhas de cache pelo protocolo MESI/MOESI (*ping-pong* de linha de cache / *false sharing*).
* **Solução Proposta:**
  1. **Alinhamento de Linha de Cache (`alignas(64)`):** Isolar os cursores atômicos e metadados críticos de vértices em estruturas alinhadas a 64 bytes para eliminar *false sharing* durante a caminhada paralela.
  2. **Reconciliação *Lock-Free* de Caminhadas Abertas:** Como todo vértice possui grau par $D$ e cada passagem intermediária consome exatamente 2 arestas ($1\text{ Vermelha} + 1\text{ Azul}$), qualquer caminhada que não feche um ciclo imediato devido à concorrência entre threads deixará um número estritamente par de extremidades abertas por vértice. Isso permite que as threads caminhem de forma **100% *lock-free*** usando apenas incremento atômico leve (`atomic capture`), seguido de uma etapa determinística $O(T)$ de costura das extremidades abertas.
  3. **Pré-divisão Vetorizada de Multiplicidades Pares:** Arestas paralelas com peso par $w(u,v) = 2q + r$ podem ter sua parte par ($q$) dividida diretamente entre os subgrafos vermelho e azul em paralelo por linha (sem contenção atômica), restando apenas o subgrafo esparso de resíduos ímpares ($r \in \{0, 1\}$) para a caminhada Euleriana.

### 3.2. Mapeamento da Árvore de Recursão na Topologia NUMA e Cache L3 (CCX/CCD)
* **Desafio Arquitetural:** Em servidores Dual-Socket do CloudLab (2 nós NUMA e múltiplos *Core Complex Dies* — CCDs — com caches L3 independentes), instanciar `#pragma omp task` sem controle de afinidade faz com que subgrafos filhos sejam alocados em um socket e processados por threads no socket oposto, gerando acessos remotos de memória (NUMA *remote hits*) de alta latência.
* **Solução Proposta:** Explorar o isomorfismo entre a **árvore binária da Divisão e Conquista ($D \to D/2 \to D/4 \dots$)** e a **hierarquia física do servidor**:
  * No nível 1 da recursão, os dois subgrafos de grau $D/2$ são completamente independentes. Utilizando afinidade de threads (`OMP_PLACES=cores`, `OMP_PROC_BIND=spread`) e política de alocação *first-touch* por subárvore, cada ramo principal da recursão fica confinado a um domínio NUMA / Socket e, nos níveis subsequentes, a fatias locais do Cache L3 (CCD), zerando o tráfego *cross-socket*.

### 3.3. Localidade Espacial de Cache (Representação *Flat CSR*) e *Cut-off* de Tarefas
* **Representação Compacta:** Substituir listas encadeadas ou `std::vector<std::vector<...>>` (que sofrem *pointer chasing* e *cache misses* em L1/L2) por uma representação contígua tipo *Compressed Sparse Row* (*Flat CSR* / *Flat Edge Array*), maximizando o aproveitamento do *hardware prefetcher*.
* **Controle de Granularidade (*Task Cut-off*):** Definir experimentalmente a profundidade crítica (`cutoff_depth` / grau mínimo $D_{\text{min}}$) a partir da qual o *overhead* do escalonador de tarefas do OpenMP supera o ganho do paralelismo intra-subgrafo, chaveando para execução sequencial local ao cache L1/L2 da thread.

---

## 4. Metodologia Experimental e Avaliação no CloudLab
O desenvolvimento funcional e a validação matemática exata serão realizados localmente, e a avaliação experimental de desempenho e microarquitetura será conduzida nos clusters do **CloudLab**, comparando três versões incrementais:
1. **`seq` (Baseline Sequencial):** Implementação sequencial otimizada do algoritmo BvN via divisão Euleriana.
2. **`omp_naive` (Paralelo Ingênuo):** Uso direto de `#pragma omp task` e cursores contíguos sem alinhamento de linha de cache, sem pré-divisão de multiplicidades e sem afinidade NUMA.
3. **`omp_arch` (Paralelo Orientado à Arquitetura):** Cursores `alignas(64)`, pré-divisão embaraçosamente paralela de multiplicidades pares, reconciliação *lock-free*, *Flat CSR*, *task cut-off* calibrado e afinidade de subárvores NUMA/Cache (`OMP_PROC_BIND`).

### Métricas Avaliadas (via `perf stat` e instrumentação):
* **Desempenho e Escalabilidade:** Tempo de execução por fase, *Speedup* forte e fraco variando de 1 até o número máximo de *cores* físicos/lógicos do nó no CloudLab, e variação de tamanho de matriz ($N \in \{128, 512, 1024, 2048\}$) e grau ($D \in \{64, 256, 1024, 4096\}$).
* **Contadores de Hardware (`perf`):**
  * Taxa de *misses* na hierarquia de cache (`L1-dcache-load-misses`, `LLC-load-misses`, `cache-misses`).
  * Instruções por Ciclo (**IPC**) e ciclos de parada por memória/coerência.
  * Impacto do *Task Cut-off* e da política de *Thread Binding* (`spread` vs. `close` vs. sem afinidade) em máquina multi-socket NUMA.

---

## 5. Cronograma de Execução
1. **Etapa 1 — Estrutura Base e Validação Sequencial:** Implementação do gerador de matrizes de tráfego (uniforme, *skewed/hotspot* e *bursty*), Fase 1 (*Padding*), Fase 2/3 Sequencial, Fase 4 (*Cleanup*) e validador matemático exato ($\sum P_k == M$).
2. **Etapa 2 — Paralelização OpenMP (`omp_naive` e `omp_arch`):** Implementação da caminhada Euleriana paralela com `atomic capture`, costura de caminhadas abertas, alinhamento `alignas(64)` e recursão via OpenMP Tasks com *cut-off*.
3. **Etapa 3 — Experimentos Arquiteturais no CloudLab:** Execução automatizada de benchmarks em nós multi-socket no CloudLab, coleta de contadores de hardware com `perf stat` e ajuste fino de *cut-off* e afinidade NUMA.
4. **Etapa 4 — Análise de Resultados e Relatório Final:** Geração dos gráficos de escalabilidade, análise de cache/NUMA e redação do relatório técnico final.

---

## 6. Referências Bibliográficas Base
1. COLE, R.; OST, K.; SCHIRRA, S. Edge-coloring bipartite multigraphs in $O(E \log D)$ time. *Combinatorica*, v. 21, p. 5-12, 2001. Disponível em: <https://dl.acm.org/doi/abs/10.5555/890218>.
2. DUFOSSÉ, F.; UÇAR, B. Notes on Birkhoff–von Neumann decomposition of doubly stochastic matrices. *Linear Algebra and its Applications*, v. 497, p. 108-115, 2016.
3. LIN, C. H. et al. Deflection-Compensated Birkhoff-von-Neumann Switches. Em: *IEEE INFOCOM 2016 - The 35th Annual IEEE International Conference on Computer Communications*. IEEE, 2016. p. 1-9.
4. CHANG, C.-S.; LEE, D.-S.; JOU, Y. S. Load balanced Birkhoff-von Neumann switches, part I: one-stage buffering. *Computer Communications*, v. 25, n. 6, p. 611-622, 2002.
5. MCKEOWN, N. The iSLIP scheduling algorithm for input-queued switches. *IEEE/ACM Transactions on Networking*, v. 7, n. 2, p. 188-201, 1999.
