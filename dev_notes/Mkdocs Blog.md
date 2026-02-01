# Mkdocs Blog

博客参考

1. [tom-jerr/tom-jerr.github.io](https://github.com/tom-jerr/tom-jerr.github.io)
2. [TonyCrane/note: TonyCrane's Public Notebook](https://github.com/TonyCrane/note)
3. [科学空间|Scientific Spaces](https://spaces.ac.cn/)

## Material

以下记录了我的 mkdocs material 的修改/学习过程

1. favicon 制作，使用了我的 github 头像，找了一个在线圆形图片裁剪，生成了圆形 favicon

2. 修改起始时间，在 `footer.html` 当中的 `startDate`

3. 在 `docs/assets/document_dates/user.config.css` 修改文档创建时间的 placement，头像不弄成灰色

4. 利用插件 `mkdocs-awesome-nav` 自动生成 navigation，写了一个 python 脚本 `helpers/generate_navs` 检查 `index.md` front matter，以及 `mkdocs.yml` 当中的相关链接是否符合预期

5. 利用 blogging 插件，在主页显示最近所写的博客

6. 设置个人链接，在 mkdocs.yml 当中的 extra: socal

7. [Admonitions - Material for MkDocs](https://squidfunk.github.io/mkdocs-material/reference/admonitions/#supported-types) 有多种格式，可以用来发布一些 announcement

8. 定制字体，在 font 下设置了 jetbrains mono 作为英文，中文的字体是在 extra css 当中引入了 lxgw 字体

9. 构建 comment 系统

   [Adding a comment system - Material for MkDocs](https://squidfunk.github.io/mkdocs-material/setup/adding-a-comment-system/?h=giscus#giscus-integration) 参考文档，新建了一个 comment 仓库，然后生成 script

10. 常用的 front matter (元数据)，写在文档顶部

    ```txt
    ---
    author: xxx
    tags: xxx
    comments: true 		# 打开评论
    hide:
    	- toc
    	- navigation 	# 隐藏导航和大纲
    nostatistics: true	# 隐藏本页数据统计
    icon: link			# 本页的 icon
    description: xxx	# 本页描述，会出现在首页的 blog 当中
    ---
    ```

11. 图像问题

    1. 图像不是默认居中的

       调整 custom.css 中的 img 完成

    2. 无法显示正确的图像位置，多了一个 GPTQ  路径

       在配置文件当中设置 `use_use_directory_urls: false`，参考 [【MkDocs踩坑】图片路径问题的排查与解决 - 萑澈的寒舍](https://hs.cnies.org/archives/mkdocs-404-imgs) [markdown - configuring image filepaths correctly in mkdocs - Stack Overflow](https://stackoverflow.com/questions/71074662/configuring-image-filepaths-correctly-in-mkdocs/71083184#71083184)

12. 公式问题

    1. 不显示换行

       无法换行这个是由于 mathjax3 引入的，无法修复。[The line break(\\) is not work · Issue #2312 · mathjax/MathJax](https://github.com/mathjax/MathJax/issues/2312)

    2. 公式渲染错误

       是因为`$$` 需要前后没有空行。另外如何将 `\displaylines` 和前后空格自动化地加入到渲染过程中，解决方法参考 [issue](https://github.com/mathjax/MathJax/issues/2312#issuecomment-2440036455)

       并且由于 typora 的 缩进还很坑，有的是 3 个缩进，缩进必须是 4 个才能正常渲染，否则在一些 List 当中公式无法渲染

       公式的规范化问题最终写了一个 `helpers/fix_math_blocks.py` 解决

13. 在 header 当中添加了 about me 链接，更改了 `overrides/partial/header.html`

14. `mdx_truly_sane_lists` 插件与公式渲染会出现冲突，我测试了 python 3.9 和 3.12 都不需要这个插件，但是 3.13 似乎需要。而且 3.9 版本会动态 reload，这个对开发很友好

15. link 可以是 pdf 文件，我放置了自己的简历

16. 在 `docs/css` 当中创建了 `mathjax.css & markdown.css` 以更改公式大小和颜色、各种行间距

## Launch with Github

根据 [Publishing your site - Material for MkDocs](https://squidfunk.github.io/mkdocs-material/publishing-your-site/?h=publish#with-github-actions)，构建了 ci，并启动 pages，一定要完成这两个步骤

另外，[Setup - Mkdocs Blogging Plugin](https://liang2kl.github.io/mkdocs-blogging-plugin/#publish-with-github-pages) 需要给 github pages 更多操作才能让 blog 显示时间正常，否则都会以最新 build 的时间作为所有的 blog 时间

## TODO

考虑像科学空间一样，添加一个引导到 kimi 的链接让其总结博客内容
