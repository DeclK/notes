# Mkdocs Blog

尝试使用 mkdocs 构建一个优雅的博客网站。计划是以 material 作为模板，在此模板之上构建自己的一些个性化，其中可以参考下别人的 plugin

学习计划很简单，先看下 material 的一些基础文档，然后看下别人的博客有没有我喜欢的设计

## Material

记录熟悉 mkdocs material 的过程，也是一些常见的需求

1. favicon 制作，使用了我的 github 头像，找了一个在线圆形图片裁剪，生成了圆形 favicon

2. 修改起始时间，在 footer.html 当中的 startDate

3. 修改文档创建时间的 placement，头像不弄成灰色

4. 自动生成 navigation，index.md

5. 利用 blogging 插件，在主页显示最近所写的博客

6. 个人链接，在 mkdocs.yml 当中的 extra: socal

7. [Admonitions - Material for MkDocs](https://squidfunk.github.io/mkdocs-material/reference/admonitions/#supported-types) 有多种格式

8. 定制字体，在 font 下设置了 jetbrains mono 作为英文，中文的字体不知道如何设置，似乎是在 extra css 当中引入了 lxgw 字体，然后在 custom css 当中正式设置了

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
       
       需要重新修改 about me 方式

12. 公式问题

    有一些公式直接不显示，无法换行，矩阵这些都不行

    无法换行这个是由于 mathjax3 引入的，无法修复。[The line break(\\) is not work · Issue #2312 · mathjax/MathJax](https://github.com/mathjax/MathJax/issues/2312)

    但是矩阵渲染仍然无法解决，发现问题，是因为`$$` 需要前后加入空格，这样才能正常渲染。另外如何将 `\displaylines` 和前后空格自动化地加入到渲染过程中

    另外如果有 indent 也是不显示的。后来发现不是公式不显示，而是一些特殊的符号引起了 html 解析的错误。这似乎是无法解决的，似乎苏神早已有所探索 [近乎完美地解决MathJax与Marked的冲突 - 科学空间|Scientific Spaces](https://spaces.ac.cn/archives/10332) 明天再看！

    最后发现是缩进插件导致的。并且由于 typora 的 缩进还很坑，有的是 3 个缩进，有的是 4 个，所以导致公式渲染出现问题

13. 嵌套列表需要额外下载  mdx_truly_sane_lists 插件，我测试了 python 3.9 和 3.12 都不需要这个插件，但是 3.13 似乎需要。而且 3.9 版本会动态 reload，这个对开发很友好。最后发现这个插件会影响公式的渲染

overrides 当中可以放置一些个性化的东西，例如 icons，pdf 也行

link 似乎也可以是图片，能够直接打开

assets 当中可以放置 pdf 文件，然后可以链接过去

## Launch with Github

先看了官方文档，构建了 ci [Publishing your site - Material for MkDocs](https://squidfunk.github.io/mkdocs-material/publishing-your-site/?h=publish#with-github-actions)

然后启动 pages

[Setup - Mkdocs Blogging Plugin](https://liang2kl.github.io/mkdocs-blogging-plugin/#publish-with-github-pages) 需要给 github pages 更多操作

## TODO

像科学空间一样，一个链接可以跳转到 kimi，然后给特定的提示符

https://www.kimi.com/_prefill_chat?prefill_prompt=你好&send_immediately=true&force_search=false&enable_reasoning=false
