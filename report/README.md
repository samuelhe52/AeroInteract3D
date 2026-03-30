# Report Workflow

本目录提供一套可复用的 final tech report 写作与导出流程。

## 文件说明

- `final-tech-report.md`
  可直接编辑的报告模板。
- `Makefile`
  本目录内的构建入口。
- `build-pdf.sh`
  使用 `pandoc + xelatex` 构建 PDF。
- `pandoc-header.tex`
  Pandoc 生成 PDF 时使用的 LaTeX 头部配置。
- `.build/`
  隐藏的 PDF 构建产物目录。
- `assets/`
  报告中引用的图片资源目录。

## 快速开始

1. 进入 `report/`
2. 编辑 `final-tech-report.md`
3. 把图片放到 `assets/`
3. 运行：

```bash
cd report
make report
```

或：

```bash
cd report
./build-pdf.sh
```

## 自定义输入输出

可以指定输入 Markdown 和输出目录：

```bash
cd report
./build-pdf.sh final-tech-report.md .build
```

## 依赖

需要本机安装：

- `pandoc`
- `xelatex`
- 中文字体 `Source Han Serif SC`
- 等宽字体 `JetBrains Mono`

当前构建参数参考了 `~/projects/Gomoku/report/` 的 PDF 工作流。
