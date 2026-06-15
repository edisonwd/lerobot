#!/bin/sh

usage() {
    cat << EOF
用法: $0 [选项] <模型组织> <模型名称>
功能: ModelScope 模型迁移为 Hugging Face 本地缓存
默认 HF 缓存目录: ~/.cache/huggingface/hub
可通过环境变量 HF_HUB_CACHE 自定义缓存根

必传参数:
  <模型组织>    模型作者/命名空间
  <模型名称>    模型名称

可选选项:
  -d DIR    ModelScope 模型上层目录 (必填)
  -c        启用复制模式(默认:软链接)
  -h        查看帮助

示例:
  $0 -d /root/gpufree-data google paligemma-3b-pt-224
  $0 -c -d /root/gpufree-data google paligemma-3b-pt-224
EOF
    exit 1
}

COPY_MODE="false"
CUSTOM_MS_ROOT=""

# 解析参数
while getopts "cd:h" opt
do
    case "$opt" in
        c) COPY_MODE="true" ;;
        d) CUSTOM_MS_ROOT="$OPTARG" ;;
        h) usage ;;
        \?) echo "错误: 无效参数 -$OPTARG" >&2; usage ;;
        :) echo "错误: 参数 -$OPTARG 需要路径值" >&2; usage ;;
    esac
done
shift $(expr $OPTIND - 1)

# 校验模型参数
if [ $# -ne 2 ]; then
    echo "错误: 请传入 模型组织、模型名称" >&2
    usage
fi
MODEL_ORG="$1"
MODEL_NAME="$2"
SNAPSHOT_HASH="local_$(date +%Y%m%d%H%M%S)"

# 必须指定 MS 根目录
if [ -z "$CUSTOM_MS_ROOT" ]; then
    echo "错误: 请使用 -d 指定 ModelScope 模型上层目录" >&2
    usage
fi

# 拼接源模型路径
MS_MODEL_DIR="${CUSTOM_MS_ROOT}/${MODEL_ORG}/${MODEL_NAME}"
if [ ! -d "$MS_MODEL_DIR" ]; then
    echo "错误: 源目录不存在 -> $MS_MODEL_DIR" >&2
    exit 1
fi

# 处理 HF 缓存路径
if [ -n "${HF_HUB_CACHE:-}" ]; then
    HF_BASE=$(eval echo "$HF_HUB_CACHE")
else
    HF_BASE="${HOME}/.cache/huggingface/hub"
fi

HF_MODEL_DIR="${HF_BASE}/models--${MODEL_ORG}--${MODEL_NAME}"
HF_SNAPSHOTS="${HF_MODEL_DIR}/snapshots"
HF_REFS="${HF_MODEL_DIR}/refs"

# 打印信息
echo "============================================="
echo "ModelScope 源路径: $MS_MODEL_DIR"
echo "HF Hub 缓存根:     $HF_BASE"
echo "HF 模型目录:       $HF_MODEL_DIR"
if [ "$COPY_MODE" = "true" ]; then
    echo "运行模式:           文件复制"
else
    echo "运行模式:           软链接"
fi
echo "快照标识:           $SNAPSHOT_HASH"
echo "============================================="

# 创建目录
mkdir -p "$HF_SNAPSHOTS" "$HF_REFS"

# 执行迁移
if [ "$COPY_MODE" = "true" ]; then
    echo "正在复制模型文件..."
    cp -r "$MS_MODEL_DIR" "${HF_SNAPSHOTS}/${SNAPSHOT_HASH}"
else
    echo "正在创建软链接..."
    ln -sf "$MS_MODEL_DIR" "${HF_SNAPSHOTS}/${SNAPSHOT_HASH}"
fi

# 写入 refs/main 【无尾换行】
printf "%s" "$SNAPSHOT_HASH" > "${HF_REFS}/main"

echo ""
echo "✅ 迁移完成！"
echo "加载代码示例:"
echo "from transformers import AutoModel"
echo "model = AutoModel.from_pretrained('${MODEL_ORG}/${MODEL_NAME}', local_files_only=True)"