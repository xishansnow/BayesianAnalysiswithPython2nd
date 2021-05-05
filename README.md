# 关于本书

A book for beginners of geographical knowledge graph.

## 使用方法 Usage

### 构建本书 Building the book

如果您要继续开发并编译BayesianAnalysiswithPython2nd，则应该：
If you'd like to develop on and build the BayesianAnalysiswithPython2nd book, you should:

- 克隆此仓库，并转到仓库根目录
- 运行`pip install -r requirements.txt`（建议您在虚拟环境中执行此操作）
- （推荐）删除现有的`BayesianAnalysiswithPython2nd/_build/`目录
-  运行`jupyter-book build BayesianAnalysiswithPython2nd/`


- Clone this repository, go to it's root directory
- Run `pip install -r requirements.txt` (it is recommended you do this within a virtual environment)
- (Recommended) Remove the existing `BayesianAnalysiswithPython2nd/_build/` directory
- Run `jupyter-book build BayesianAnalysiswithPython2nd/`

完整的HTML版本数据将创建在`BayesianAnalysiswithPython2nd/_build/html/`文件夹内。
A fully-rendered HTML version of the book will be built in `BayesianAnalysiswithPython2nd/_build/html/`.

### 发布本书 Hosting the book

本书的HTML版本位于仓库的 `gh-pages` 分支上。 仓库已经创建了GitHub actions工作流，该工作流会根据 `master` 分支的推送或拉取请求自动编译书籍并将其推送到gh-pages分支。
The html version of the book is hosted on the `gh-pages` branch of this repo. A GitHub actions workflow has been created that automatically builds and pushes the book to this branch on a push or pull request to main.

如果您希望禁用此自动化，可以删除GitHub action工作流（存储在.github目录内)，并按在编译完成后，使用下述流程发布该书至 `gh-papges` 分支 ：
If you wish to disable this automation, you may remove the GitHub actions workflow and build the book manually by:

- 进入本地仓库的根目录 
- 运行`ghp-import -n -p -f BayesianAnalysiswithPython2nd/_build/html`
  
- Navigating to your local build; and running,
- `ghp-import -n -p -f BayesianAnalysiswithPython2nd/_build/html`

这将自动将您的构建推送到gh-pages分支。 有关此托管过程的更多信息，请参见[here](https://jupyterbook.org/publish/gh-pages.html#manually-host-your-book-with-github-pages)。
This will automatically push your build to the `gh-pages` branch. More information on this hosting process can be found [here](https://jupyterbook.org/publish/gh-pages.html#manually-host-your-book-with-github-pages).

## 贡献者 Contributors

我们欢迎并感谢您的所有贡献。 您可以在[贡献者标签](https://github.com/xishansnow/BayesianAnalysiswithPython2nd/graphs/contributors)中查看当前贡献者的列表。
We welcome and recognize all contributions. You can see a list of current contributors in the [contributors tab](https://github.com/xishansnow/BayesianAnalysiswithPython2nd/graphs/contributors).

## 感谢 Credits

该项目使用出色的开源[Jupyter Book项目](https://jupyterbook.org/)和[executablebooks/cookiecutter-jupyter-book模板](https://github.com/executablebooks/cookiecutter-jupyter)创建）。
This project is created using the excellent open source [Jupyter Book project](https://jupyterbook.org/) and the [executablebooks/cookiecutter-jupyter-book template](https://github.com/executablebooks/cookiecutter-jupyter-book).