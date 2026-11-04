"""Parallel-BvN: nó bare-metal multi-socket para benchmarks de arquitetura (NUMA, cache, perf).

Aloca um único nó físico (sem virtualização), instala as dependências,
compila o `bvn_bench` em modo Release e deixa o ambiente pronto para
`perf stat` e `numactl`. O repositório é clonado pelo CloudLab em
`/local/repository`.

Instructions:
Depois que o nó ficar pronto (o setup leva poucos minutos), acompanhe o log com
`tail -f /local/logs/bvn-setup.log`. O binário fica em
`/local/repository/build/bvn_bench`. Para rodar a suíte completa:

    cd /local/repository && ./scripts/run_cloudlab_bench.sh

Os resultados são gravados em `/local/repository/results_*`.
"""

import geni.portal as portal
import geni.rspec.pg as pg

pc = portal.Context()

pc.defineParameter(
    "hwtype", "Tipo de hardware",
    portal.ParameterType.NODETYPE, "c6420",
    longDescription=(
        "Tipo de nó físico. O padrão c6420 (Clemson) é um Dual-Socket Intel Xeon Gold 6142 "
        "com 384 GB de RAM, o mesmo usado nos resultados do trabalho. Mudar o tipo muda a "
        "topologia NUMA e torna os números não comparáveis."
    ),
)

pc.defineParameter(
    "image", "Imagem de disco",
    portal.ParameterType.IMAGE, "urn:publicid:IDN+emulab.net+image+emulab-ops//UBUNTU22-64-STD",
    longDescription="Ubuntu 22.04 (GCC 11, CMake 3.22), a mesma versão usada nos experimentos.",
)

pc.defineParameter(
    "runBench", "Rodar a suíte de benchmarks ao final do setup",
    portal.ParameterType.BOOLEAN, False,
    longDescription=(
        "Se marcado, executa scripts/run_cloudlab_bench.sh logo após compilar. "
        "Pode levar bastante tempo; o progresso fica em /local/logs/bvn-bench.log."
    ),
)

pc.defineParameter(
    "perfGovernor", "Fixar o governor da CPU em performance",
    portal.ParameterType.BOOLEAN, True,
    advanced=True,
    longDescription="Reduz a variação entre repetições evitando escalonamento de frequência.",
)

params = pc.bindParameters()

if not params.hwtype:
    pc.reportError(portal.ParameterError("Informe um tipo de hardware.", ["hwtype"]))
pc.verifyParameters()

request = pc.makeRequestRSpec()

node = request.RawPC("node-0")
node.hardware_type = params.hwtype
node.disk_image = params.image

setup_cmd = "sudo /local/repository/scripts/cloudlab/setup.sh {bench} {gov}".format(
    bench="--run-bench" if params.runBench else "",
    gov="--perf-governor" if params.perfGovernor else "",
)
node.addService(pg.Execute(shell="bash", command=setup_cmd))

pc.printRequestRSpec(request)
