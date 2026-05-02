#!/bin/bash
# ============================================================
# sync_to_server.sh
# 在服务器端执行，把 /data1/lixiang/EmotionalLLM/ 按本地 2026-04-18
# 重构后的结构对齐。
#
# 用法：
#   scp this file 到服务器，或 ssh 后 cat > sync_to_server.sh
#   cd /data1/lixiang/EmotionalLLM
#   bash sync_to_server.sh
#
# 安全：
#   - 所有操作均为 mv/rmdir，不删除任何实验数据
#   - 执行前备份：  tar czf /tmp/emollm_pre_restructure.tar.gz --exclude='code/*/result' --exclude='observation_v3' .
#   - 如需回滚，每步都有对应反向 mv 命令（注释中）
# ============================================================

set -e
cd /data1/lixiang/EmotionalLLM

echo "=== 阶段 A: 清理与归档 ==="

# 1. 删除 code/ 根下两个冗余 cross_eval 脚本（本地已无，服务器端可能也没有；存在则删）
[ -f code/white_box_opens2s_cross_eval.py ] && rm -v code/white_box_opens2s_cross_eval.py || echo "  (skip) code/white_box_opens2s_cross_eval.py not present"
[ -f code/white_box_voxtral_cross_eval.py ]  && rm -v code/white_box_voxtral_cross_eval.py  || echo "  (skip) code/white_box_voxtral_cross_eval.py not present"

# 2. PREVIOUS/ → archive/
[ -d PREVIOUS ] && mv -v PREVIOUS archive || echo "  (skip) PREVIOUS not present"

# 3. observation/ + observation_v2/ → archive/observation_early_docs/{observation_v1,observation_v2}
mkdir -p archive/observation_early_docs
[ -d observation ]    && mv -v observation    archive/observation_early_docs/observation_v1 || true
[ -d observation_v2 ] && mv -v observation_v2 archive/observation_early_docs/observation_v2 || true

# 4. code/white_box_opens2s_v1/ → archive/
[ -d code/white_box_opens2s_v1 ] && mv -v code/white_box_opens2s_v1 archive/white_box_opens2s_v1 || true

# 5. finalpaper2/ → archive/pipid_paper_template/
[ -d finalpaper2 ] && mv -v finalpaper2 archive/pipid_paper_template || true

# 6. LATEST/ → reports/
[ -d LATEST ] && mv -v LATEST reports || true

echo "=== 阶段 B: 顶层重组 ==="

# 7. paper/ → refs/（先做，因为 finalpaper 要占用 paper 这个名）
[ -d paper ] && mv -v paper refs || true

# 8. finalpaper/ → paper/
[ -d finalpaper ] && mv -v finalpaper paper || true

# 9. dataset/ → data/
[ -d dataset ] && mv -v dataset data || true

# 10. observation_v3/ → results/observation_v3/
mkdir -p results
[ -d observation_v3 ] && mv -v observation_v3 results/observation_v3 || true

# 11. docs/：meeting/ + 框架.png + temp/research.md
mkdir -p docs
[ -d meeting ] && mv -v meeting docs/meeting || true
[ -f 框架.png ] && mv -v 框架.png docs/framework.png || true
[ -f temp/research.md ] && mv -v temp/research.md docs/related_work_research.md || true
# temp/ 清理（仅在为空时删除）
[ -d temp ] && rmdir --ignore-fail-on-non-empty temp && echo "  temp/ removed (was empty)" || true

echo "=== 检查结果 ==="
echo "顶层目录："
ls -la | grep ^d | awk '{print "  "$NF}'
echo ""
echo "提醒：如果上面有任何「not present」或 skip，说明该项已不需要操作（可能本地未镜像或已执行过）。"
echo "完成！"
