(window.webpackJsonp=window.webpackJsonp||[]).push([[168],{247:function(e,t,n){"use strict";n.r(t),n.d(t,"frontMatter",(function(){return c})),n.d(t,"metadata",(function(){return s})),n.d(t,"toc",(function(){return l})),n.d(t,"default",(function(){return p}));var r=n(3),a=n(8),i=(n(0),n(270)),o=n(277),c={id:"nevergrad_sweeper",title:"Nevergrad Sweeper plugin",sidebar_label:"Nevergrad Sweeper plugin"},s={unversionedId:"plugins/nevergrad_sweeper",id:"plugins/nevergrad_sweeper",isDocsHomePage:!1,title:"Nevergrad Sweeper plugin",description:"PyPI",source:"@site/docs/plugins/nevergrad_sweeper.md",slug:"/plugins/nevergrad_sweeper",permalink:"/docs/next/plugins/nevergrad_sweeper",editUrl:"https://github.com/facebookresearch/hydra/edit/master/website/docs/plugins/nevergrad_sweeper.md",version:"current",lastUpdatedBy:"Jasha10",lastUpdatedAt:1622629068,formattedLastUpdatedAt:"6/2/2021",sidebar_label:"Nevergrad Sweeper plugin",sidebar:"docs",previous:{title:"Ax Sweeper plugin",permalink:"/docs/next/plugins/ax_sweeper"},next:{title:"Optuna Sweeper plugin",permalink:"/docs/next/plugins/optuna_sweeper"}},l=[{value:"Installation",id:"installation",children:[]},{value:"Usage",id:"usage",children:[]},{value:"Example of training using Nevergrad hyperparameter search",id:"example-of-training-using-nevergrad-hyperparameter-search",children:[]},{value:"Defining the parameters",id:"defining-the-parameters",children:[{value:"Defining through commandline override",id:"defining-through-commandline-override",children:[]},{value:"Defining through config file",id:"defining-through-config-file",children:[]}]}],u={toc:l};function p(e){var t=e.components,n=Object(a.a)(e,["components"]);return Object(i.b)("wrapper",Object(r.a)({},u,n,{components:t,mdxType:"MDXLayout"}),Object(i.b)("p",null,Object(i.b)("a",{parentName:"p",href:"https://pypi.org/project/hydra-nevergrad-sweeper/"},Object(i.b)("img",{parentName:"a",src:"https://img.shields.io/pypi/v/hydra-nevergrad-sweeper",alt:"PyPI"})),"\n",Object(i.b)("img",{parentName:"p",src:"https://img.shields.io/pypi/l/hydra-nevergrad-sweeper",alt:"PyPI - License"}),"\n",Object(i.b)("img",{parentName:"p",src:"https://img.shields.io/pypi/pyversions/hydra-nevergrad-sweeper",alt:"PyPI - Python Version"}),"\n",Object(i.b)("a",{parentName:"p",href:"https://pypistats.org/packages/hydra-nevergrad-sweeper"},Object(i.b)("img",{parentName:"a",src:"https://img.shields.io/pypi/dm/hydra-nevergrad-sweeper.svg",alt:"PyPI - Downloads"})),Object(i.b)(o.a,{text:"Example application",to:"plugins/hydra_nevergrad_sweeper/example",mdxType:"ExampleGithubLink"}),Object(i.b)(o.a,{text:"Plugin source",to:"plugins/hydra_nevergrad_sweeper",mdxType:"ExampleGithubLink"})),Object(i.b)("p",null,Object(i.b)("a",{parentName:"p",href:"https://facebookresearch.github.io/nevergrad/"},"Nevergrad")," is a derivative-free optimization platform providing a library of state-of-the-art algorithms for hyperparameter search. This plugin provides Hydra applications a mechanism to use ",Object(i.b)("a",{parentName:"p",href:"https://facebookresearch.github.io/nevergrad/"},"Nevergrad")," algorithms to optimize experiment/application parameters."),Object(i.b)("h3",{id:"installation"},"Installation"),Object(i.b)("pre",null,Object(i.b)("code",{parentName:"pre",className:"language-commandline"},"pip install hydra-nevergrad-sweeper --upgrade\n")),Object(i.b)("h3",{id:"usage"},"Usage"),Object(i.b)("p",null,"Once installed, add ",Object(i.b)("inlineCode",{parentName:"p"},"hydra/sweeper=nevergrad")," to your command. Alternatively, override ",Object(i.b)("inlineCode",{parentName:"p"},"hydra/sweeper")," in your config:"),Object(i.b)("pre",null,Object(i.b)("code",{parentName:"pre",className:"language-yaml"},"defaults:\n  - override hydra/sweeper: nevergrad\n")),Object(i.b)("p",null,"The default configuration is ",Object(i.b)(o.b,{to:"plugins/hydra_nevergrad_sweeper/hydra_plugins/hydra_nevergrad_sweeper/config.py",mdxType:"GithubLink"},"here"),".\nThere are several standard approaches for configuring plugins. Check ",Object(i.b)("a",{parentName:"p",href:"../patterns/configuring_plugins"},"this page")," for more information."),Object(i.b)("h2",{id:"example-of-training-using-nevergrad-hyperparameter-search"},"Example of training using Nevergrad hyperparameter search"),Object(i.b)("p",null,"We include an example of how to use this plugin. The file ",Object(i.b)(o.b,{to:"plugins/hydra_nevergrad_sweeper/example/my_app.py",mdxType:"GithubLink"},"example/my_app.py")," implements an example of minimizing a (dummy) function using a mixture of continuous and discrete parameters."),Object(i.b)("p",null,"You can discover the Nevergrad sweeper parameters with:"),Object(i.b)("pre",null,Object(i.b)("code",{parentName:"pre",className:"language-yaml",metastring:'title="$ python your_app hydra/sweeper=nevergrad --cfg hydra -p hydra.sweeper"',title:'"$',python:!0,your_app:!0,"hydra/sweeper":"nevergrad","--cfg":!0,hydra:!0,"-p":!0,'hydra.sweeper"':!0},"# @package hydra.sweeper\n_target_: hydra_plugins.hydra_nevergrad_sweeper.core.NevergradSweeper\noptim:\n  optimizer: NGOpt\n  budget: 80\n  num_workers: 10\n  noisy: false\n  maximize: false\n  seed: null\nparametrization: {}\nversion: 1\n")),Object(i.b)("p",null,"The function decorated with ",Object(i.b)("inlineCode",{parentName:"p"},"@hydra.main()")," returns a float which we want to minimize, the minimum is 0 and reached for:"),Object(i.b)("pre",null,Object(i.b)("code",{parentName:"pre",className:"language-yaml"},"db: mnist\nlr: 0.12\ndropout: 0.33\nbatch_size=4\n")),Object(i.b)("p",null,"To run hyperparameter search and look for the best parameters for this function, clone the code and run the following command in the ",Object(i.b)("inlineCode",{parentName:"p"},"plugins/hydra_nevergrad_sweeper")," directory:"),Object(i.b)("pre",null,Object(i.b)("code",{parentName:"pre",className:"language-bash"},"python example/my_app.py -m\n")),Object(i.b)("p",null,"You can also override the search space parametrization:"),Object(i.b)("pre",null,Object(i.b)("code",{parentName:"pre",className:"language-bash"},"python example/my_app.py --multirun db=mnist,cifar batch_size=4,8,16 \\\n'lr=tag(log, interval(0.001, 1))' 'dropout=interval(0,1)'\n")),Object(i.b)("p",null,"The initialization of the sweep and the first 5 evaluations (out of 100) look like this:"),Object(i.b)("pre",null,Object(i.b)("code",{parentName:"pre",className:"language-text"},"[2020-10-08 20:13:53,592][HYDRA] NevergradSweeper(optimizer=NGOpt, budget=100, num_workers=10) minimization\n[2020-10-08 20:13:53,593][HYDRA] with parametrization Dict(batch_size=Choice(choices=Tuple(4,8,16),weights=Array{(1,3)}),db=Choice(choices=Tuple(mnist,cifar),weights=Array{(1,2)}),dropout=Scalar{Cl(0,1,b)}[sigma=Log{exp=2.0}],lr=Log{exp=3.162277660168379,Cl(0.001,1,b)}):{'db': 'mnist', 'lr': 0.03162277660168379, 'dropout': 0.5, 'batch_size': 8}\n[2020-10-08 20:13:53,593][HYDRA] Sweep output dir: multirun/2020-10-08/20-13-53\n[2020-10-08 20:13:55,023][HYDRA] Launching 10 jobs locally\n[2020-10-08 20:13:55,023][HYDRA]        #0 : db=mnist lr=0.03162277660168379 dropout=0.5 batch_size=16\n[2020-10-08 20:13:55,217][__main__][INFO] - dummy_training(dropout=0.500, lr=0.032, db=mnist, batch_size=16) = 13.258\n[2020-10-08 20:13:55,218][HYDRA]        #1 : db=cifar lr=0.018178519762066934 dropout=0.5061074452336254 batch_size=4\n[2020-10-08 20:13:55,408][__main__][INFO] - dummy_training(dropout=0.506, lr=0.018, db=cifar, batch_size=4) = 0.278\n[2020-10-08 20:13:55,409][HYDRA]        #2 : db=cifar lr=0.10056825918734161 dropout=0.6399687427725211 batch_size=4\n[2020-10-08 20:13:55,595][__main__][INFO] - dummy_training(dropout=0.640, lr=0.101, db=cifar, batch_size=4) = 0.329\n[2020-10-08 20:13:55,596][HYDRA]        #3 : db=mnist lr=0.06617542958182834 dropout=0.5059497416026679 batch_size=8\n[2020-10-08 20:13:55,812][__main__][INFO] - dummy_training(dropout=0.506, lr=0.066, db=mnist, batch_size=8) = 5.230\n[2020-10-08 20:13:55,813][HYDRA]        #4 : db=mnist lr=0.16717013388679514 dropout=0.6519070394318255 batch_size=4\n...\n[2020-10-08 20:14:27,988][HYDRA] Best parameters: db=cifar lr=0.11961221693764439 dropout=0.37285878409770895 batch_size=4\n")),Object(i.b)("p",null,"and the final 2 evaluations look like this:"),Object(i.b)("pre",null,Object(i.b)("code",{parentName:"pre",className:"language-text"},"[HYDRA]     #8 : db=mnist batch_size=4 lr=0.094 dropout=0.381\n[__main__][INFO] - my_app.py(dropout=0.381, lr=0.094, db=mnist, batch_size=4) = 1.077\n[HYDRA]     #9 : db=mnist batch_size=4 lr=0.094 dropout=0.381\n[__main__][INFO] - my_app.py(dropout=0.381, lr=0.094, db=mnist, batch_size=4) = 1.077\n[HYDRA] Best parameters: db=mnist batch_size=4 lr=0.094 dropout=0.381\n")),Object(i.b)("p",null,"The run also creates an ",Object(i.b)("inlineCode",{parentName:"p"},"optimization_results.yaml")," file in your sweep folder with the parameters recommended by the optimizer:"),Object(i.b)("pre",null,Object(i.b)("code",{parentName:"pre",className:"language-yaml"},"best_evaluated_result: 0.381\n\nbest_evaluated_params:\n  batch_size: 4\n  db: mnist\n  dropout: 0.381\n  lr: 0.094\n\nname: nevergrad\n")),Object(i.b)("h2",{id:"defining-the-parameters"},"Defining the parameters"),Object(i.b)("p",null,"The plugin supports two types of parameters: ",Object(i.b)("a",{parentName:"p",href:"https://facebookresearch.github.io/nevergrad/parametrization_ref.html#nevergrad.p.Choice"},"Choices")," and ",Object(i.b)("a",{parentName:"p",href:"https://facebookresearch.github.io/nevergrad/parametrization_ref.html#nevergrad.p.Scalar"},"Scalars"),". They can be defined either through config file or commandline override."),Object(i.b)("h3",{id:"defining-through-commandline-override"},"Defining through commandline override"),Object(i.b)("p",null,"Hydra provides a override parser that support rich syntax. More documentation can be found in (",Object(i.b)("a",{parentName:"p",href:"/docs/next/advanced/override_grammar/basic"},"OverrideGrammer/Basic"),") and (",Object(i.b)("a",{parentName:"p",href:"/docs/next/advanced/override_grammar/extended"},"OverrideGrammer/Extended"),"). We recommend you go through them first before proceeding with this doc."),Object(i.b)("h4",{id:"choices"},"Choices"),Object(i.b)("p",null,"To override a field with choices:"),Object(i.b)("pre",null,Object(i.b)("code",{parentName:"pre",className:"language-commandline"},"'key=1,5'\n'key=shuffle(range(1, 8))'\n'key=range(1,5)'\n")),Object(i.b)("p",null,"You can tag an override with ",Object(i.b)("inlineCode",{parentName:"p"},"ordered")," to indicate it's a ",Object(i.b)("a",{parentName:"p",href:"https://facebookresearch.github.io/nevergrad/parametrization_ref.html#nevergrad.p.TransitionChoice"},Object(i.b)("inlineCode",{parentName:"a"},"TransitionChoice"))),Object(i.b)("pre",null,Object(i.b)("code",{parentName:"pre",className:"language-commandline"},"`key=tag(ordered, choice(1,2,3))`\n")),Object(i.b)("h4",{id:"scalar"},"Scalar"),Object(i.b)("pre",null,Object(i.b)("code",{parentName:"pre",className:"language-commandline"},"`key=interval(1,12)`             # Interval are float by default\n`key=int(interval(1,8))`         # Scalar bounds cast to a int\n`key=tag(log, interval(1,12))`   # call ng.p.Log if tagged with log\n")),Object(i.b)("h3",{id:"defining-through-config-file"},"Defining through config file"),Object(i.b)("h4",{id:"choices-1"},"Choices"),Object(i.b)("p",null,"Choices are defined with a list in a config file."),Object(i.b)("pre",null,Object(i.b)("code",{parentName:"pre",className:"language-yaml"},"db:\n  - mnist\n  - cifar\n")),Object(i.b)("h4",{id:"scalars"},"Scalars"),Object(i.b)("p",null,"Scalars can be defined in config files, with fields:"),Object(i.b)("ul",null,Object(i.b)("li",{parentName:"ul"},Object(i.b)("inlineCode",{parentName:"li"},"init"),": optional initial value"),Object(i.b)("li",{parentName:"ul"},Object(i.b)("inlineCode",{parentName:"li"},"lower")," : optional lower bound"),Object(i.b)("li",{parentName:"ul"},Object(i.b)("inlineCode",{parentName:"li"},"upper"),": optional upper bound"),Object(i.b)("li",{parentName:"ul"},Object(i.b)("inlineCode",{parentName:"li"},"log"),": set to ",Object(i.b)("inlineCode",{parentName:"li"},"true")," for log distributed values"),Object(i.b)("li",{parentName:"ul"},Object(i.b)("inlineCode",{parentName:"li"},"step"),": optional step size for looking for better parameters. In linear mode, this is an additive step; in logarithmic mode it is multiplicative."),Object(i.b)("li",{parentName:"ul"},Object(i.b)("inlineCode",{parentName:"li"},"integer"),": set to ",Object(i.b)("inlineCode",{parentName:"li"},"true")," for integers (favor floats over integers whenever possible)")),Object(i.b)("p",null,"Providing only ",Object(i.b)("inlineCode",{parentName:"p"},"lower")," and ",Object(i.b)("inlineCode",{parentName:"p"},"upper")," bound will set the initial value to the middle of the range and the step to a sixth of the range.\n",Object(i.b)("strong",{parentName:"p"},"Note"),": unbounded scalars (scalars with no upper and/or lower bounds) can only be defined through a config file."))}p.isMDXComponent=!0},270:function(e,t,n){"use strict";n.d(t,"a",(function(){return p})),n.d(t,"b",(function(){return m}));var r=n(0),a=n.n(r);function i(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function o(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function c(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?o(Object(n),!0).forEach((function(t){i(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):o(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function s(e,t){if(null==e)return{};var n,r,a=function(e,t){if(null==e)return{};var n,r,a={},i=Object.keys(e);for(r=0;r<i.length;r++)n=i[r],t.indexOf(n)>=0||(a[n]=e[n]);return a}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(r=0;r<i.length;r++)n=i[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(a[n]=e[n])}return a}var l=a.a.createContext({}),u=function(e){var t=a.a.useContext(l),n=t;return e&&(n="function"==typeof e?e(t):c(c({},t),e)),n},p=function(e){var t=u(e.components);return a.a.createElement(l.Provider,{value:t},e.children)},d={inlineCode:"code",wrapper:function(e){var t=e.children;return a.a.createElement(a.a.Fragment,{},t)}},b=a.a.forwardRef((function(e,t){var n=e.components,r=e.mdxType,i=e.originalType,o=e.parentName,l=s(e,["components","mdxType","originalType","parentName"]),p=u(n),b=r,m=p["".concat(o,".").concat(b)]||p[b]||d[b]||i;return n?a.a.createElement(m,c(c({ref:t},l),{},{components:n})):a.a.createElement(m,c({ref:t},l))}));function m(e,t){var n=arguments,r=t&&t.mdxType;if("string"==typeof e||r){var i=n.length,o=new Array(i);o[0]=b;var c={};for(var s in t)hasOwnProperty.call(t,s)&&(c[s]=t[s]);c.originalType=e,c.mdxType="string"==typeof e?e:r,o[1]=c;for(var l=2;l<i;l++)o[l]=n[l];return a.a.createElement.apply(null,o)}return a.a.createElement.apply(null,n)}b.displayName="MDXCreateElement"},271:function(e,t,n){"use strict";function r(e){return!0===/^(\w*:|\/\/)/.test(e)}function a(e){return void 0!==e&&!r(e)}n.d(t,"b",(function(){return r})),n.d(t,"a",(function(){return a}))},272:function(e,t,n){"use strict";var r=n(0),a=n.n(r),i=n(11),o=n(271),c=n(7),s=Object(r.createContext)({collectLink:function(){}}),l=n(273),u=function(e,t){var n={};for(var r in e)Object.prototype.hasOwnProperty.call(e,r)&&t.indexOf(r)<0&&(n[r]=e[r]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols){var a=0;for(r=Object.getOwnPropertySymbols(e);a<r.length;a++)t.indexOf(r[a])<0&&Object.prototype.propertyIsEnumerable.call(e,r[a])&&(n[r[a]]=e[r[a]])}return n};t.a=function(e){var t,n,p,d=e.isNavLink,b=e.to,m=e.href,g=e.activeClassName,h=e.isActive,f=e["data-noBrokenLinkCheck"],v=e.autoAddBaseUrl,O=void 0===v||v,y=u(e,["isNavLink","to","href","activeClassName","isActive","data-noBrokenLinkCheck","autoAddBaseUrl"]),j=Object(l.b)().withBaseUrl,w=Object(r.useContext)(s),_=b||m,N=Object(o.a)(_),D=null==_?void 0:_.replace("pathname://",""),x=void 0!==D?(n=D,O&&function(e){return e.startsWith("/")}(n)?j(n):n):void 0,A=Object(r.useRef)(!1),C=d?i.e:i.c,k=c.default.canUseIntersectionObserver;Object(r.useEffect)((function(){return!k&&N&&window.docusaurus.prefetch(x),function(){k&&p&&p.disconnect()}}),[x,k,N]);var P=null!==(t=null==x?void 0:x.startsWith("#"))&&void 0!==t&&t,z=!x||!N||P;return x&&N&&!P&&!f&&w.collectLink(x),z?a.a.createElement("a",Object.assign({href:x},_&&!N&&{target:"_blank",rel:"noopener noreferrer"},y)):a.a.createElement(C,Object.assign({},y,{onMouseEnter:function(){A.current||(window.docusaurus.preload(x),A.current=!0)},innerRef:function(e){var t,n;k&&e&&N&&(t=e,n=function(){window.docusaurus.prefetch(x)},(p=new window.IntersectionObserver((function(e){e.forEach((function(e){t===e.target&&(e.isIntersecting||e.intersectionRatio>0)&&(p.unobserve(t),p.disconnect(),n())}))}))).observe(t))},to:x||""},d&&{isActive:h,activeClassName:g}))}},273:function(e,t,n){"use strict";n.d(t,"b",(function(){return i})),n.d(t,"a",(function(){return o}));var r=n(10),a=n(271);function i(){var e=Object(r.default)().siteConfig,t=(e=void 0===e?{}:e).baseUrl,n=void 0===t?"/":t,i=e.url;return{withBaseUrl:function(e,t){return function(e,t,n,r){var i=void 0===r?{}:r,o=i.forcePrependBaseUrl,c=void 0!==o&&o,s=i.absolute,l=void 0!==s&&s;if(!n)return n;if(n.startsWith("#"))return n;if(Object(a.b)(n))return n;if(c)return t+n;var u=n.startsWith(t)?n:t+n.replace(/^\//,"");return l?e+u:u}(i,n,e,t)}}}function o(e,t){return void 0===t&&(t={}),(0,i().withBaseUrl)(e,t)}},274:function(e,t,n){try{e.exports=n(275)}catch(a){var r={};e.exports={useAllDocsData:function(){return r},useActivePluginAndVersion:function(){}}}},275:function(e,t,n){"use strict";Object.defineProperty(t,"__esModule",{value:!0}),t.useDocVersionSuggestions=t.useActiveDocContext=t.useActiveVersion=t.useLatestVersion=t.useVersions=t.useActivePluginAndVersion=t.useActivePlugin=t.useDocsData=t.useAllDocsData=void 0;var r=n(23),a=n(38),i=n(276);t.useAllDocsData=function(){return a.useAllPluginInstancesData("docusaurus-plugin-content-docs")},t.useDocsData=function(e){return a.usePluginData("docusaurus-plugin-content-docs",e)},t.useActivePlugin=function(e){void 0===e&&(e={});var n=t.useAllDocsData(),a=r.useLocation().pathname;return i.getActivePlugin(n,a,e)},t.useActivePluginAndVersion=function(e){void 0===e&&(e={});var n=t.useActivePlugin(e),a=r.useLocation().pathname;if(n)return{activePlugin:n,activeVersion:i.getActiveVersion(n.pluginData,a)}},t.useVersions=function(e){return t.useDocsData(e).versions},t.useLatestVersion=function(e){var n=t.useDocsData(e);return i.getLatestVersion(n)},t.useActiveVersion=function(e){var n=t.useDocsData(e),a=r.useLocation().pathname;return i.getActiveVersion(n,a)},t.useActiveDocContext=function(e){var n=t.useDocsData(e),a=r.useLocation().pathname;return i.getActiveDocContext(n,a)},t.useDocVersionSuggestions=function(e){var n=t.useDocsData(e),a=r.useLocation().pathname;return i.getDocVersionSuggestions(n,a)}},276:function(e,t,n){"use strict";Object.defineProperty(t,"__esModule",{value:!0}),t.getDocVersionSuggestions=t.getActiveDocContext=t.getActiveVersion=t.getLatestVersion=t.getActivePlugin=void 0;var r=n(23);t.getActivePlugin=function(e,t,n){void 0===n&&(n={});var a=Object.entries(e).find((function(e){e[0];var n=e[1];return!!r.matchPath(t,{path:n.path,exact:!1,strict:!1})})),i=a?{pluginId:a[0],pluginData:a[1]}:void 0;if(!i&&n.failfast)throw new Error("Can't find active docs plugin for pathname="+t+", while it was expected to be found. Maybe you tried to use a docs feature that can only be used on a docs-related page? Existing docs plugin paths are: "+Object.values(e).map((function(e){return e.path})).join(", "));return i},t.getLatestVersion=function(e){return e.versions.find((function(e){return e.isLast}))},t.getActiveVersion=function(e,n){var a=t.getLatestVersion(e);return[].concat(e.versions.filter((function(e){return e!==a})),[a]).find((function(e){return!!r.matchPath(n,{path:e.path,exact:!1,strict:!1})}))},t.getActiveDocContext=function(e,n){var a,i,o=t.getActiveVersion(e,n),c=null==o?void 0:o.docs.find((function(e){return!!r.matchPath(n,{path:e.path,exact:!0,strict:!1})}));return{activeVersion:o,activeDoc:c,alternateDocVersions:c?(a=c.id,i={},e.versions.forEach((function(e){e.docs.forEach((function(t){t.id===a&&(i[e.name]=t)}))})),i):{}}},t.getDocVersionSuggestions=function(e,n){var r=t.getLatestVersion(e),a=t.getActiveDocContext(e,n),i=a.activeVersion!==r;return{latestDocSuggestion:i?null==a?void 0:a.alternateDocVersions[r.name]:void 0,latestVersionSuggestion:i?r:void 0}}},277:function(e,t,n){"use strict";n.d(t,"b",(function(){return l})),n.d(t,"a",(function(){return u}));var r=n(3),a=n(0),i=n.n(a),o=n(272),c=n(10),s=n(274);function l(e){return i.a.createElement(o.a,Object(r.a)({},e,{to:(t=e.to,a=Object(s.useActiveVersion)(),Object(c.default)().siteConfig.customFields.githubLinkVersionToBaseUrl[null!==(n=null==a?void 0:a.name)&&void 0!==n?n:"current"]+t),target:"_blank"}));var t,n,a}function u(e){var t,n=null!==(t=e.text)&&void 0!==t?t:"Example";return i.a.createElement(l,e,i.a.createElement("span",null,"\xa0"),i.a.createElement("img",{src:"https://img.shields.io/badge/-"+n+"-informational",alt:"Example"}))}}}]);