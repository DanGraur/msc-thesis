#!/bin/sh
echo "date,time,pid,app,proc_perc,mem_perc" >> $2 | top -b -d 1 -p $1 | awk -v OFS="," '$1+0>0 { print strftime("%Y-%m-%d,%H:%M:%S"),$1,$NF,$9,$10; fflush() }' >> $2
