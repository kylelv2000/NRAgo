#!/bin/bash
THREAD_NUM=6
workers=20

#指定分隔符
IFS='
'
#定义描述符为9的FIFO管道
mkfifo tmp
exec 9<>tmp
rm -f tmp
 
#预先写入指定数量的空格符，一个空格符代表一个进程
for ((i=0;i<$THREAD_NUM;i++))
do
    echo >&9
done

function parallel() {
    for folder in $@/*;do 
        if [ ! -d results$folder ];then
            mkdir results$folder
        fi
        state=-1
        for file in $folder/*;do
            if test -f $file;then
                state=0
                break
            fi
            if test -d $file;then
                state=1
                break
            fi
        done
        if [ $state -eq 0 ];then
            for i in $folder/*; do
                echo $i
                # timeout 1800 python3 src/nra_go.py $i > results$i -W $workers
                read -u9
                {
                    start=$[$(date +%s%N)/1000000]
                    timeout 1800 python3 src/nra_go.py $i > results$i -W $workers
                    end=$[$(date +%s%N)/1000000]
                    take=$(( end - start ))
                    echo $i , ${take} ms >> times.csv
                    sleep 1
                    #控制进程数：一个任务完成后，写入一个空格字符到管道，新的任务将可以执行
                    echo >&9
                }&
            done
        fi
        if [ $state -eq 1 ];then
            parallel $folder
        fi
    done
}

parallel $1

wait
echo "\n全部任务执行结束"
