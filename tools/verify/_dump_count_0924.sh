#!/bin/bash
cd /home/fedsa/DyGLib
echo "== npz 清单 =="
ls -1 results/samples_dump/*/*.npz 2>/dev/null | wc -l
ls -1 results/samples_dump/*/*.npz 2>/dev/null
echo "== queue.log 尾部 6 行（截断显示） =="
tail -6 tools/queue/queue.log | cut -c1-150
echo "== 每数据集 npz 数 =="
for ds in WikiVote RedditHyperlinkTitle RedditHyperlinkBody BitcoinAlpha BitcoinOTC; do
  echo "$ds: $(ls results/samples_dump/$ds/*.npz 2>/dev/null | wc -l)"
done
