(window.webpackJsonp=window.webpackJsonp||[]).push([[9],{54:function(e,r,n){"use strict";n.r(r),n.d(r,"frontMatter",(function(){return u})),n.d(r,"rightToc",(function(){return c})),n.d(r,"default",(function(){return l}));n(0);var t=n(82);function a(){return(a=Object.assign||function(e){for(var r=1;r<arguments.length;r++){var n=arguments[r];for(var t in n)Object.prototype.hasOwnProperty.call(n,t)&&(e[t]=n[t])}return e}).apply(this,arguments)}function o(e,r){if(null==e)return{};var n,t,a=function(e,r){if(null==e)return{};var n,t,a={},o=Object.keys(e);for(t=0;t<o.length;t++)n=o[t],r.indexOf(n)>=0||(a[n]=e[n]);return a}(e,r);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(t=0;t<o.length;t++)n=o[t],r.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(a[n]=e[n])}return a}var u={id:"workdir",title:"Customizing working directory pattern",sidebar_label:"Customizing working directory pattern"},c=[],i={rightToc:c},p="wrapper";function l(e){var r=e.components,n=o(e,["components"]);return Object(t.b)(p,a({},i,n,{components:r,mdxType:"MDXLayout"}),Object(t.b)("p",null,"See the ",Object(t.b)("a",a({parentName:"p"},{href:"intro"}),"intro")," for details about how to apply the customization."),Object(t.b)("p",null,"Run output directory grouped by day:"),Object(t.b)("pre",null,Object(t.b)("code",a({parentName:"pre"},{className:"language-yaml"}),"hydra:\n  run:\n    dir: ./outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}\n")),Object(t.b)("p",null,"Sweep sub directory contains the override parameters for the job instance:"),Object(t.b)("pre",null,Object(t.b)("code",a({parentName:"pre"},{className:"language-yaml"}),"hydra:\n  sweep:\n    subdir: ${hydra.job.num}_${hydra.job.id}_${hydra.job.override_dirname}\n")),Object(t.b)("p",null,"Run output directory grouped by job name:"),Object(t.b)("pre",null,Object(t.b)("code",a({parentName:"pre"},{className:"language-yaml"}),"hydra:\n  run:\n    dir: outputs/${hydra.job.name}/${now:%Y-%m-%d_%H-%M-%S}\n")),Object(t.b)("p",null,"Run output directory can contain user configuration variables:"),Object(t.b)("pre",null,Object(t.b)("code",a({parentName:"pre"},{className:"language-yaml"}),"hydra:\n  run:\n    dir: outputs/${now:%Y-%m-%d_%H-%M-%S}/opt:${optimizer.type}\n\n")),Object(t.b)("p",null,"Run output directory can contain override parameters for the job"),Object(t.b)("pre",null,Object(t.b)("code",a({parentName:"pre"},{className:"language-yaml"}),"hydra:\n  run:\n    dir: output/${hydra.job.override_dirname}\n")))}l.isMDXComponent=!0},82:function(e,r,n){"use strict";n.d(r,"a",(function(){return c})),n.d(r,"b",(function(){return b}));var t=n(0),a=n.n(t),o=a.a.createContext({}),u=function(e){var r=a.a.useContext(o),n=r;return e&&(n="function"==typeof e?e(r):Object.assign({},r,e)),n},c=function(e){var r=u(e.components);return a.a.createElement(o.Provider,{value:r},e.children)};var i="mdxType",p={inlineCode:"code",wrapper:function(e){var r=e.children;return a.a.createElement(a.a.Fragment,{},r)}},l=Object(t.forwardRef)((function(e,r){var n=e.components,t=e.mdxType,o=e.originalType,c=e.parentName,i=function(e,r){var n={};for(var t in e)Object.prototype.hasOwnProperty.call(e,t)&&-1===r.indexOf(t)&&(n[t]=e[t]);return n}(e,["components","mdxType","originalType","parentName"]),l=u(n),b=t,d=l[c+"."+b]||l[b]||p[b]||o;return n?a.a.createElement(d,Object.assign({},{ref:r},i,{components:n})):a.a.createElement(d,Object.assign({},{ref:r},i))}));function b(e,r){var n=arguments,t=r&&r.mdxType;if("string"==typeof e||t){var o=n.length,u=new Array(o);u[0]=l;var c={};for(var p in r)hasOwnProperty.call(r,p)&&(c[p]=r[p]);c.originalType=e,c[i]="string"==typeof e?e:t,u[1]=c;for(var b=2;b<o;b++)u[b]=n[b];return a.a.createElement.apply(null,u)}return a.a.createElement.apply(null,n)}l.displayName="MDXCreateElement"}}]);