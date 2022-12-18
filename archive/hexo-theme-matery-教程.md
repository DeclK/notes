---
title: hexo-theme-matery 教程
tags:
  - Hexo
  - 博客搭建
categories:
  - 软件
  - Hexo
abbrlink: b129f5ae
date: 2021-07-02 16:46:32
---
# hexo-theme-matery note

## 配置

参考原 github 项目进行整理：https://github.com/blinkfox/hexo-theme-matery

### _config.yml 的修改建议

在 hexo 根目录下的 _config.yml 文件中：

1. 切换 theme 为 hexo-theme-matery

2. url 改为网站 url (如：http://xxx.github.io)

3. per_page 数值改为6及6的倍数，这样文章列表在各个屏幕下都能较好显示

4. 插入图片（有点复杂，可以先跳过）。根据 [知乎链接](https://zhuanlan.zhihu.com/p/265077468) 配置好 post_asset_folder，里面提到的 typora 技巧也很实用，建议采用。且一般来讲链接中提到的 hexo-renderer-marked 插件都是已经内置好的，不然你的博客不会渲染成功。

   但是这个方法不能控制图片大小，而且配置过 typora 后还要再改图片的路径。我再下载了 hexo-asset-image 插件过后，直接用 typora 中的 html 方法 `<img src='post_name/img.jpg'> `引用图片，不需要二次更改路径

### 新建分类 categories 页

```shell
sudo hexo new page 'categories'
```

categories 页是用来展示所有分类的页面

修改 `/source/categories/index.md` 文件的 front-matter (Front-matter 是文件最上方以 --- 分隔的区域，用于指定个别文件的变量)

```shell
---
title: categories
date: 2018-09-30 17:25:30
type: "categories"
layout: "categories"
---
```

关于 page 和 post 的区别：两者其实很相似，你可以把 page 看作是特殊的 post，用来放置一些特殊内容：如分类、标签等等。

hexo 的分类是有等级关系的，我们以下面的 post front-matter 为例

```markdown
title: 我的博客搭建笔记
date: 2021-06-29 15:26:41
tags:
- hexo
- 博客搭建
categories:
- 软件基础
- hexo
```

这个文件在分类时会被分配到所有提到的 categories 中，这些 categories 的路径是逐渐往下的 `/categories/软件基础/hexo/`，它们都有自己的 page 来收纳属于自己类别的文章。

### 新建标签 tags 页

新建 page

```shell
hexo new page "tags"
```

修改对应 index.md

```shell
---
title: tags
date: 2018-09-30 18:23:38
type: "tags"
layout: "tags"
---
```

给 post 指定 tags 只需要在 front-matter 中写好就行了，如上面 categories 的例子。

### 新建关于我 about 页

### 新建留言板 contact 页

### 新建友情链接 friends 页

新建方法都是和上面的方法一样的

## 菜单导航配置与渲染

### 名称、路径、图标

配置菜单导航的名称、路径url、和图标icon，配置文件在 `themes/hexo-theme-matery/._config.yml`

更改 menu 部分即可

```shell
menu:
  # 把 Index 改为 Home
  Home:
    url: /
    icon: fas fa-home
  Tags:
    url: /tags
    icon: fas fa-tags
```

可该的部分：

1. 名称可以是中文
2. 图标可以在 [Font Awesome](https://fontawesome.com/icons) 中查找
3. 还可以使用二级菜单。二级菜单在实现上可以理解为创建了一个软链接，到你指定的 page 上。

由于我对于这个导航还是比较满意，就不作过多修改

### 修改页脚

页脚信息可能需要做定制化修改，修改的地方在主题文件的 `/layout/_partial/footer.ejs` 文件中，包括站点、使用的主题、访问量等。

由于代码看不明白，找到了一个比较详细的 [教程](https://sunhwee.com/posts/6e8839eb.html)。根据教程修改了版权信息，增加了网站运行时间，访问人数的代码目前不需要修改

### 代码高亮

修改 Hexo 根目录下 `_config.yml` 文件中 `highlight.enable` 的值为 `false`，并将 `prismjs.enable` 的值设置为 `true`

以上的设置并不管用，没有高亮也没有行号

但是我下载了 hexo-prism-plugin 又卸载掉这个插件，就有高亮了，但依旧没有行号

考虑是 matery.css 文件中 pre 下 paddidng 不够大的问题，增加 padding 但依然没能解决

考虑是不是本身 prism 文件出了问题，重新到官网上下载了 prism.js 和 prism.css 替换原来系统中对应的文件，同时调整上面提到的 padding 参数，最后成功！

prism.js 文件位置在 `node_modules/prismjs`

prism.css 文件位置在 `themes/hexo-theme-matery/source/libs/prism`

现在想要去除隔离的那条竖线，尝试重复 [github issue](https://github.com/blinkfox/hexo-theme-matery/issues/103) 的操作，操作过后竖线没有去除，不过稍微调整了一下代码位置，也挺好看

最终，我通过修改 prism.css 中 border-right 为 0px，去除了行号和代码之间的分隔线！

```css
.line-numbers .line-numbers-rows {
	position: absolute;
	pointer-events: none;
	top: -0.2em;
	font-size: 100%;
	left: -3.8em;
	width: 3em; /* works for line-numbers below 1000 lines */
	letter-spacing: -1px;
	border-right: 0px solid #999;
```

现在解决标号和 code 没对齐的问题，修改 prism.css 文件中 `.line-numbers .line-numbers-rows ` 下 top 参数为 -0.2em，微调成功！治好了强迫症！

### 渲染表格

hexo 对表格的渲染与 typora 是不一样的，正文和表格需要间隔两行。而且 hexo 并不会显示双括号，建议使用代码格式引用双括号 `{{}}`

## 配置 theme 中的 _config.yml

1. 根据配置文件中的注释，简单修改了下面的设置

   dream, music, video, recommend, github & social link, reward, clicklove, myProjects, mySkills, subtitle, banner

2. 取消 rainbow 特效

   ```css
   .bg-cover:after {
       -webkit-animation: none;
       animation: none;
   }
   ```

3. 在 hexo d 时遇到问题

   ```shell
   err: Error: Spawn failed
   ```

   第二天自动好了，根本原因是网络没有走通的问题，之后可能再次遇到，到时候再解决，建议在 github 上搜索

   现在发现了，由于我使用的是 root 用户下的 git 配置，而我 root 用户下 git config 没有设置好参数所以部署失败。我在 root 目录的 .gitconfig 和 .ssh 下配置好了 user.name ssh 等必要 git 配置就成功了。这说明了在 linux 下不同的用户都需要自己去配置 git

4. 修改博客 feature image 只选用1张简单图来代表。原来有 24 张图，我把24个路径全部改为同1张图。

5. 因为之后需要写入数学公式，将 mathjax 改为 true。但发现显示公式渲染不正常，而且小括号显示不出来。根据 [github issue](https://github.com/blinkfox/hexo-theme-matery/issues/119) 解决无法换行问题，将渲染器换为 hexo-renderer-kramed 而且代码高亮插件似乎并没受到影响。根据 [MathJax常见问题](https://adaning.github.io/posts/33457.html) 解决小括号无法显示问题。根据 [博客](https://www.jianshu.com/p/7ab21c7f0674) 再进一步配置 kramed 渲染，解决大括号的冲突问题。也遇到过本地 `hexo s` 能够渲染公式，但是 `hexo d` 部署后公式无法渲染的情况，根据 [博客](https://blog.csdn.net/qq_44846324/article/details/114582328) 中解决语义冲突解决问题

## 配置 matery.css

为了进一步设置我们的网页，让其更具有个性化，就需要进一步调整 matery.css 文件。

1. 设置 导航颜色

   ```css
   .bg-color {
       background-image: linear-gradient(to right, #BEBEBE 0%, #708090 100%);
   }
   ```

   查询颜色代码网址：https://tool.oschina.net/commons?type=3

2. 由于刚开始加载时 banner 图片没有迅速加载，会默认先加载橙色，我改成灰色

   修改“\Hexo\themes\hexo-theme-matery\layout\ _partial\index-cover.ejs"文件中的第63行即可。

   ```ejs
   <div class="carousel-item red white-text bg-cover about-cover">
   ```

   把“red”修改为其他颜色即可。我改为 slate gray

3. 改变 progress-bar 颜色

4. 改变回到顶部按钮颜色 top-scroll

5. 改变封面打字效果的文字大小、颜色 .bg-cover .description

6. 修改了打字效果的颜色过后，我发现文章标题的颜色也跟着改了。解决方法是在下面的 .bg-cover .description 增加属性 color: #color_code，这样就能分别调整它们的颜色。

7. 修改 about 页面链接颜色 aboutme

8. 修改 archive 页面时间线颜色 cd-timeline

## 插件优化

### 文章 url 优化

我没有使用项目中主要介绍的转拼音方法，那样生成的链接太长了，而是使用 hexo-abbrlink 插件

```sehll
npm install hexo-abbrlink --save
```

在根目录 _config.yml 文件中修改

```yaml
permalink: archives/:abbrlink.html
abbrlink:
    alg: crc32   #算法： crc16(default) and crc32
    rep: hex     #进制： dec(default) and hex
```

在 hexo 三连时遇到报错

```shell
FATAL YAMLException: duplicated mapping key (111:1)
```

说明有原配置文件中已经有 permalink 的定义，我们需要把里本身的 permalink 代码注释掉，完美运行！现在我们的文章 url 最后是特殊的数字id `/archives/48732.html`

### 文章字数统计插件

### 添加 emoji 表情支持

### 搜索

上面三个部分都是按照 hexo-thme-matery github 项目配置的

### 评论插件utterance

matery 在配置文件中告诉我们，有的评论软件有安全隐患，推荐使用 utterance

尝试 utterance，如果成功，则返回卸载 valine，注销 lean cloud

也尝试了 livere 安装也很不友好

成功在 contact 页面下展示了 utterance，方法是在 contact.ejs 文档下找了一个地方插入下面的代码（记得改为 r自己的 github repo，形式为 <OWNER>/<NAME> ），

```ejs
<script src="https://utteranc.es/client.js"
repo="[ENTER REPO HERE]"
issue-term="pathname"
theme="github-light"
crossorigin="anonymous"
async>
</script>
```

我插入在 `<div class="card">` 后面，是管用的。因为原项目说，插入到你的 layout 需要出现的地方，我不懂前端的代码，只能胡乱插入了，管用就行！我猜测这是一个“卡片类”能够存放你的内容，以白色卡片在页面中展示出来。

用同样的方法在 post 下添加评论。找到 `/hexo-theme-matery/layout/_partial/post-detail.ejs` 文件，找到评论的布局之处（搜索主题自带的评论插件如 gittalk，就能找到）插入上面的代码即可。但是渲染过后背景是透明的，不太好看，我希望想 contact 一样有白色背景，那就复制一下 contact.ejs 中的“卡片类”评论区就行了。

### 分类优化

matery 主题的分类没有多层分类，在 categories 页面只有像 tag 一样标签，这样的分类又有什么用呢？根据 [Hexo Matery 主题添加多级分类](https://notes.zhangxiaocai.cn/posts/5a99eb4d.html#toc-heading-8) 进行设置，可以得到更好的分类页，类别之间将有层次关系

具体来讲，采用了博客里最新更新的代码，又作了以下改动：

1. 给 category-item, category-count 等增加 color 属性，改为自己喜欢的颜色

2. 给 category-title 增加 font-size 属性，修改标题大小

3. 由于每次进入目录默认要展开一个类别，改动下方代码，让所有类别初始状态默认折叠

   ```js
   /* origin code: <li  class="<%= subCats.length > 0 ? 'active' : '' %>" > */
   <li  class="<%= subCats.length > 10000 ? 'active' : '' %>" >
   ```

4. 删除箭头变化 ` category-item-action col s11 m11`

## TODO

### 网站SEO优化

Search Engine Optimization

将自己的网站提交给搜索引擎

### 不断更新

持续更新博客内容，完善分类