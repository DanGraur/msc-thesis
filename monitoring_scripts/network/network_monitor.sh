#!/usr/bin/env bash
#
# Check bandwidth plugin for Nagios
#
# Usage: check_bandwidth.sh [-i interface] [-s sleep] [-w warning] [-c critical]
#     -i, --interface         Interface name (eth0 by default)
#     -s, --sleep             Sleep time between both statistics measures
#     -w, --warning           Warning value (KB/s)
#     -c, --critical          Critical value (KB/s)
#     -h, --help              Display this screen
#
# (c) 2014, Benjamin Dos Santos <benjamin.dossantos@gmail.com>
# https://github.com/bdossantos/nagios-plugins
#
# Modified by Dan Graur
#

interface=$1
sleep=${sleep:=1}

if [[ ! -f "/sys/class/net/${interface}/statistics/rx_bytes" ]] ||
  [[ ! -f "/sys/class/net/${interface}/statistics/tx_bytes" ]]; then
  echo "CRITICAL - Could not fetch '${interface}' interface statistics"
  exit 2
fi

# This might need to be run asynchronously using '&' due to the sleep, although I am not sure
echo "date,time,interface,tx_kbps,rx_kbps" > $2
while :
do
  rx1=$(cat "/sys/class/net/${interface}/statistics/rx_bytes")
  tx1=$(cat "/sys/class/net/${interface}/statistics/tx_bytes")
  sleep "$sleep"
  rx2=$(cat "/sys/class/net/${interface}/statistics/rx_bytes")
  tx2=$(cat "/sys/class/net/${interface}/statistics/tx_bytes")

  tx_bps=$((tx2 - tx1))
  rx_bps=$((rx2 - rx1))
  tx_kbps=$((tx_bps / 1024))
  rx_kbps=$((rx_bps / 1024))

  current_date=$(date +%Y-%m-%d,%H:%M:%S)
  status="${current_date},${interface},${tx_kbps},${rx_kbps}"
  echo $status >> $2
done

exit ${exit_status:=3}