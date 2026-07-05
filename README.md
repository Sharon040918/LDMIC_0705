# LDMIC 多输入图像压缩复现

本项目基于论文 **LDMIC: Learning-based Distributed Multi-view Image Coding** 的代码进行复现和适配，当前版本用于本地 `C1`、`C2` 两个文件夹上的三输入滑动窗口训练与评估。

当前数据读取方式为：从 `C1`、`C2` 两个目录中分别使用滑动窗口构造 3 张输入图像，用于训练 `Multi_LDMIC` 模型。

## 代码目录

建议上传代码和配置文件：

- `models/`
- `lib/`
- `train.py`
- `train_multi.py`
- `eval.py`
- `eval_multi.py`
- `mydatapro.py`
- `requirements.txt`
- `README.md`
- `LICENSE`
- `.gitignore`


## 环境配置

创建 conda 环境：

```bash
conda create -n ldmic python=3.10 -y
conda activate ldmic
```

安装依赖：

```bash
pip install -r requirements.txt
```

Windows 下安装 `compressai` 时可能需要 Microsoft C++ Build Tools。如果 `compressai` 编译失败，需要安装 Visual Studio Build Tools，并勾选“使用 C++ 的桌面开发”。

安装完成后可用以下命令检查环境：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import compressai; print(compressai.__version__)"
python -c "from models.ldmic import Multi_LDMIC; print('model import ok')"
```

## 数据目录

数据集不放入 Git 仓库，建议按以下格式放置：

```text
data/
  C1/
    00000000.png
    00000005.png
    ...
  C2/
    00000000.png
    00000005.png
    ...
```

当前 `mydatapro.py` 中的主要设置为：

- `num_camera=3`：每个样本输入 3 张图像。
- `dir_num=2`：只使用 `C1` 和 `C2` 两个文件夹。
- 使用滑动窗口构造样本。

例如，一个样本可能由同一文件夹内连续 3 张图像组成。

## 快速训练

用于验证流程是否能跑通的小规模训练：

```bash
python train_multi.py -d ./data --data-name . --model-name Multi_LDMIC --metric mse --lambda 2048 --num-camera 3 --batch-size 1 --test-batch-size 1 --epochs 1 --num-workers 0 --patch-size 64 64 --save
```

CPU 下相对可接受的较大规模训练：

```bash
python train_multi.py -d ./data --data-name . --model-name Multi_LDMIC --metric mse --lambda 2048 --num-camera 3 --batch-size 1 --test-batch-size 1 --epochs 20 --num-workers 0 --patch-size 128 128 --save
```

模型权重默认保存在：

```text
checkpoints/mse/Multi_LDMIC/lamda2048/train-run*/
```

常见权重文件包括：

```text
ckpt.pth.tar
ckpt.best.pth.tar
```

## 模型评估

使用训练好的权重进行评估：

```bash
python eval_multi.py -d ./data --data-name multi_wildtrack --IFrameModel Multi_LDMIC --i_model_path "checkpoints/mse/Multi_LDMIC/lamda2048/train-run1/ckpt.best.pth.tar" --output ./results --num-camera 3 --entropy-estimation --crop
```

评估结果会保存到：

```text
results/
```

输出文件为 JSON 格式，主要指标包括：

- `psnr-float`：3 个输入的平均 PSNR。
- `ms-ssim-float`：3 个输入的平均 MS-SSIM。
- `bpp`：平均码率估计。
- `index0-*`、`index1-*`、`index2-*`：每个输入位置对应的单独指标。

## 参数说明

- `--lambda`：率失真权重。数值越大，通常 PSNR 越高，但 bpp 也会更高。
- `--patch-size`：训练时随机裁剪的图像块大小。图像块越大，上下文信息越多，但训练更慢。
- `--num-camera`：每个样本输入的图像数量。当前设置为 3。
- `--crop`：评估时裁剪图像尺寸，使其与模型的下采样/上采样比例对齐，建议评估时开启。
- `--entropy-estimation`：使用概率估计方式计算 bpp。

## 说明

当前评估脚本只保存指标 JSON，不保存重建图像。如果需要保存模型重建后的图片，需要在 `eval_multi.py` 中额外添加保存 `out["x_hat"]` 的逻辑。
