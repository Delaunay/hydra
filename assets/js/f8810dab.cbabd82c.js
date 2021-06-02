(window.webpackJsonp=window.webpackJsonp||[]).push([[183],{262:function(e,a,n){"use strict";n.r(a),n.d(a,"frontMatter",(function(){return p})),n.d(a,"metadata",(function(){return o})),n.d(a,"toc",(function(){return i})),n.d(a,"default",(function(){return s}));var t=n(3),r=n(8),c=(n(0),n(270)),p={id:"app_packaging",title:"Application packaging",sidebar_label:"Application packaging"},o={unversionedId:"advanced/app_packaging",id:"version-0.11/advanced/app_packaging",isDocsHomePage:!1,title:"Application packaging",description:"You can package your Hydra application along with its configuration.",source:"@site/versioned_docs/version-0.11/advanced/packaging.md",slug:"/advanced/app_packaging",permalink:"/docs/0.11/advanced/app_packaging",editUrl:"https://github.com/facebookresearch/hydra/edit/master/website/versioned_docs/version-0.11/advanced/packaging.md",version:"0.11",lastUpdatedBy:"Jasha10",lastUpdatedAt:1622629068,formattedLastUpdatedAt:"6/2/2021",sidebar_label:"Application packaging",sidebar:"version-0.11/docs",previous:{title:"Colorlog plugin",permalink:"/docs/0.11/plugins/colorlog"},next:{title:"Config search path",permalink:"/docs/0.11/advanced/search_path"}},i=[],l={toc:i};function s(e){var a=e.components,n=Object(r.a)(e,["components"]);return Object(c.b)("wrapper",Object(t.a)({},l,n,{components:a,mdxType:"MDXLayout"}),Object(c.b)("p",null,"You can package your Hydra application along with its configuration.\nThere is a working example ",Object(c.b)("a",{parentName:"p",href:"https://github.com/facebookresearch/hydra/tree/0.11_branch/examples/advanced/hydra_app_example"},"here"),"."),Object(c.b)("p",null,"You can run it with:"),Object(c.b)("pre",null,Object(c.b)("code",{parentName:"pre",className:"language-yaml"},"$ python examples/advanced/hydra_app_example/hydra_app/main.py\ndataset:\n  name: imagenet\n  path: /datasets/imagenet\n")),Object(c.b)("p",null,"To install it, use:"),Object(c.b)("pre",null,Object(c.b)("code",{parentName:"pre",className:"language-text"},"$ pip install examples/advanced/hydra_app_example\n...\nSuccessfully installed hydra-app-0.1\n")),Object(c.b)("p",null,"Run the installed app with:"),Object(c.b)("pre",null,Object(c.b)("code",{parentName:"pre",className:"language-yaml"},"$ hydra_app\ndataset:\n  name: imagenet\n  path: /datasets/imagenet\n")),Object(c.b)("p",null,"The installed app will use the packaged configuration files."))}s.isMDXComponent=!0},270:function(e,a,n){"use strict";n.d(a,"a",(function(){return d})),n.d(a,"b",(function(){return b}));var t=n(0),r=n.n(t);function c(e,a,n){return a in e?Object.defineProperty(e,a,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[a]=n,e}function p(e,a){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var t=Object.getOwnPropertySymbols(e);a&&(t=t.filter((function(a){return Object.getOwnPropertyDescriptor(e,a).enumerable}))),n.push.apply(n,t)}return n}function o(e){for(var a=1;a<arguments.length;a++){var n=null!=arguments[a]?arguments[a]:{};a%2?p(Object(n),!0).forEach((function(a){c(e,a,n[a])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):p(Object(n)).forEach((function(a){Object.defineProperty(e,a,Object.getOwnPropertyDescriptor(n,a))}))}return e}function i(e,a){if(null==e)return{};var n,t,r=function(e,a){if(null==e)return{};var n,t,r={},c=Object.keys(e);for(t=0;t<c.length;t++)n=c[t],a.indexOf(n)>=0||(r[n]=e[n]);return r}(e,a);if(Object.getOwnPropertySymbols){var c=Object.getOwnPropertySymbols(e);for(t=0;t<c.length;t++)n=c[t],a.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(r[n]=e[n])}return r}var l=r.a.createContext({}),s=function(e){var a=r.a.useContext(l),n=a;return e&&(n="function"==typeof e?e(a):o(o({},a),e)),n},d=function(e){var a=s(e.components);return r.a.createElement(l.Provider,{value:a},e.children)},u={inlineCode:"code",wrapper:function(e){var a=e.children;return r.a.createElement(r.a.Fragment,{},a)}},g=r.a.forwardRef((function(e,a){var n=e.components,t=e.mdxType,c=e.originalType,p=e.parentName,l=i(e,["components","mdxType","originalType","parentName"]),d=s(n),g=t,b=d["".concat(p,".").concat(g)]||d[g]||u[g]||c;return n?r.a.createElement(b,o(o({ref:a},l),{},{components:n})):r.a.createElement(b,o({ref:a},l))}));function b(e,a){var n=arguments,t=a&&a.mdxType;if("string"==typeof e||t){var c=n.length,p=new Array(c);p[0]=g;var o={};for(var i in a)hasOwnProperty.call(a,i)&&(o[i]=a[i]);o.originalType=e,o.mdxType="string"==typeof e?e:t,p[1]=o;for(var l=2;l<c;l++)p[l]=n[l];return r.a.createElement.apply(null,p)}return r.a.createElement.apply(null,n)}g.displayName="MDXCreateElement"}}]);