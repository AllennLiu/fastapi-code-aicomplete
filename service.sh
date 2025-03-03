#!/bin/bash

LOOP_KEEP=True
SERVE_PORT=7860
SERVE_HOST=0.0.0.0
SERVE_HTTPS=false

# SSH client public key
SSH_PUB_KEY='ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQCxrbvTeCCQvOvMQqh98MPuJxpNlAwYQUueGrY1Z3byoNuR1bThjSAq9DGG6ANuRzrHDtxPXRxURQensNJdmKN0s37tpyvbvYV5Zjg0xUWgTpP+7QCzPGzXdsONZ6CR7cUL3phClMVnUFhERZ56gU+CqBHpFJskT9Qf2nxlTPf+1UFwlDag21Vi756u81wXyMUYs2GNQjVSnCF/5U92CSsNqENifxfEDdCyCmqTm9FntCH/wT8eHL0earjGPM4Jr83QtXjncxIwoqpSkrOAPq7s/0fSKrnYbb+RfMKKyIt5dxCM0HCNgfoDaVYuwp0fu5ujuR3Prdy3ert9UTMWp9/e2iMJsskb3O3nP3I45fO8vOWF9vX0ZM+Ok/pWIPJlBY52jyKaTqU/QiqXGqoqs0XKhQnyfPn3gQQL/Py/0Kzsf4FP2zkoQhKRBRXpISU/4y5g/bpary5LBCZqmG7GlB/+98B337FJMR3nZHZXm1aBns+ElqDZiM4ix5jC7WipchSUW5RWV3RRhkqX9KrS0WdrhFGovzm22QseUwNJul7ZnSsYf6WiScGEAh5rZywfr0ZriAww65g9Vv/s47Wx4lbX3mlyjwFMSIUZkf4L3Prs2rQSelDTVs4zRxKWa9ZKOmNDe8YmrvK+LIQ/NX6CQB0wEmLHUBBstewdNBccXRy9Rw== ieciec070168@IPT-070168-HP'

# The number of workers formula: ( 2 x CPU_CORES ) + 1
CPU_CORE_NUM=$(python -c 'from multiprocessing import cpu_count; print(cpu_count())')
WORKER_NUM=$(( ( 2 * CPU_CORE_NUM ) + 1 ))
# Override system automated detected CPU cores for saving cluster resource

function usage
{
    more << EOF
Usage: $0 [Option] argv

FastAPI backend service manager.

Options:
    -s, --ssh            serve with SSH server only
    --kill, --terminate  terminate existing web process immediately
    --stag, --test       serve environ with deployment
    --prod, --main       serve environ with production
    --chat               enable chat model
    --code               enable code model
    --multi-modal        enable multi modal model
    --ssl-active         enable HTTPs protocol with SSL CA files (server.crt, server.key)

EOF
    exit 0
}

function config_sshd
{
    # release limitation of log watch
    test -z "$(grep -F fs.inotify.max_user_watches=524288 /etc/sysctl.conf)" && \
        echo fs.inotify.max_user_watches=524288 | tee -a /etc/sysctl.conf && sysctl -p

    # configure SSH settings
    echo 'root:111111' | chpasswd
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
    service ssh start || true
    ssh-keygen -t rsa -N "" -f /root/.ssh/id_rsa <<< y
    echo "$SSH_PUB_KEY" > /root/.ssh/authorized_keys
}

function keyboard_interrupt
{
    LOOP_KEEP=False
}

function server_forever
{
    echo "Interrupted with CTRL+C to exited web service automatically."
    while [ "$LOOP_KEEP" == "True" ]; do sleep 1; done
}

function main
{
    config_sshd
    if [ "${SERVE_HTTPS,,}" == "true" ]; then
        test ! -f "server.crt" -o ! -f "server.key" && \
            echo "SSL CA files server.crt or server.key not found." && exit 1
        local ssl_para_line="--certfile=server.crt --keyfile=server.key"
    fi
    test -z "$MODE_CHATBOT" -a -z "$MODE_CODE" -a -z "$MODE_MULTIMODAL" && \
        echo "Please specify an option for enabling at least one of model." && exit 1
    echo "Clearing model prediction core dump files." && rm -f core.*
    gunicorn app.main:app -n black-milan -b ${SERVE_HOST}:$SERVE_PORT \
        -t 600 --graceful-timeout 600 --keep-alive 600 $ssl_para_line \
        --worker-connections 1 --reload -w $WORKER_NUM -k uvicorn.workers.UvicornWorker
    exit 0
}

# parse arguments
while [ -n "$1" ]
do
    case $1 in
        -h|--help)
            usage
            ;;
        -s|--ssh)
            config_sshd
            server_forever
            ;;
        --kill|--terminate)
            ps -ef | grep -v grep | grep -E "$0|uvicorn" | awk '{print $2}' | grep -vE '^1$' | xargs -i kill -9 {}
            exit 0
            ;;
        --stag|--test)
            export FASTAPI_ENV=stag
            WORKER_NUM=2
            ;;
        --prod|--main)
            export FASTAPI_ENV=prod
            WORKER_NUM=8
            ;;
        --chat)
            export MODE_CHATBOT=true
            ;;
        --code)
            export MODE_CODE=true
            ;;
        --multi-modal)
            export MODE_MULTIMODAL=true
            ;;
        --ssl-active)
            export SERVE_HTTPS=true
            ;;
        * ) echo "Invalid arguments, try '-h/--help' for more information."
            exit 1
            ;;
    esac
    shift
done

# set terminate func
trap keyboard_interrupt 2

main
