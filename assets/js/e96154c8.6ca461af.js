(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[1297],{3905:function(e,t,n){"use strict";n.d(t,{Zo:function(){return p},kt:function(){return f}});var r=n(7294);function a(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function o(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function i(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?o(Object(n),!0).forEach((function(t){a(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):o(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function l(e,t){if(null==e)return{};var n,r,a=function(e,t){if(null==e)return{};var n,r,a={},o=Object.keys(e);for(r=0;r<o.length;r++)n=o[r],t.indexOf(n)>=0||(a[n]=e[n]);return a}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(r=0;r<o.length;r++)n=o[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(a[n]=e[n])}return a}var s=r.createContext({}),d=function(e){var t=r.useContext(s),n=t;return e&&(n="function"==typeof e?e(t):i(i({},t),e)),n},p=function(e){var t=d(e.components);return r.createElement(s.Provider,{value:t},e.children)},u={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},c=r.forwardRef((function(e,t){var n=e.components,a=e.mdxType,o=e.originalType,s=e.parentName,p=l(e,["components","mdxType","originalType","parentName"]),c=d(n),f=a,m=c["".concat(s,".").concat(f)]||c[f]||u[f]||o;return n?r.createElement(m,i(i({ref:t},p),{},{components:n})):r.createElement(m,i({ref:t},p))}));function f(e,t){var n=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var o=n.length,i=new Array(o);i[0]=c;var l={};for(var s in t)hasOwnProperty.call(t,s)&&(l[s]=t[s]);l.originalType=e,l.mdxType="string"==typeof e?e:a,i[1]=l;for(var d=2;d<o;d++)i[d]=n[d];return r.createElement.apply(null,i)}return r.createElement.apply(null,n)}c.displayName="MDXCreateElement"},4348:function(e,t,n){"use strict";n.r(t),n.d(t,{frontMatter:function(){return l},contentTitle:function(){return s},metadata:function(){return d},toc:function(){return p},default:function(){return c}});var r=n(2122),a=n(9756),o=(n(7294),n(3905)),i=["components"],l={id:"defaults_list_interpolation",title:"Defaults List interpolation"},s=void 0,d={unversionedId:"upgrades/1.0_to_1.1/defaults_list_interpolation",id:"upgrades/1.0_to_1.1/defaults_list_interpolation",isDocsHomePage:!1,title:"Defaults List interpolation",description:"The defaults lists are used to compose the final config object.",source:"@site/docs/upgrades/1.0_to_1.1/defaults_list_interpolation_changes.md",sourceDirName:"upgrades/1.0_to_1.1",slug:"/upgrades/1.0_to_1.1/defaults_list_interpolation",permalink:"/docs/next/upgrades/1.0_to_1.1/defaults_list_interpolation",editUrl:"https://github.com/facebookresearch/hydra/edit/master/website/docs/upgrades/1.0_to_1.1/defaults_list_interpolation_changes.md",version:"current",lastUpdatedBy:"Olivier Delalleau",lastUpdatedAt:1628009903,formattedLastUpdatedAt:"8/3/2021",frontMatter:{id:"defaults_list_interpolation",title:"Defaults List interpolation"},sidebar:"docs",previous:{title:"Defaults List Overrides",permalink:"/docs/next/upgrades/1.0_to_1.1/defaults_list_override"},next:{title:"Changes to Package Header",permalink:"/docs/next/upgrades/1.0_to_1.1/changes_to_package_header"}},p=[{value:"Migration examples",id:"migration-examples",children:[]}],u={toc:p};function c(e){var t=e.components,n=(0,a.Z)(e,i);return(0,o.kt)("wrapper",(0,r.Z)({},u,n,{components:t,mdxType:"MDXLayout"}),(0,o.kt)("p",null,"The defaults lists are used to compose the final config object.\nHydra supports a limited form of interpolation in the defaults list.\nThe interpolation style described there is deprecated in favor of a cleaner style more\nappropriate to recursive default lists."),(0,o.kt)("h2",{id:"migration-examples"},"Migration examples"),(0,o.kt)("p",null,"For example, the following snippet from Hydra 1.0 or older: "),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-yaml"},"defaults:\n  - dataset: imagenet\n  - model: alexnet\n  - dataset_model: ${defaults.0.dataset}_${defaults.1.model}\n")),(0,o.kt)("p",null,"Changes to this in Hydra 1.1 or newer:"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-yaml"},"defaults:\n  - dataset: imagenet\n  - model: alexnet\n  - dataset_model: ${dataset}_${model}\n")),(0,o.kt)("p",null,"The new style is more compact and does not require specifying the exact index of the element in the defaults list.\nThis is enables interpolating using config group values that are coming from recursive defaults."),(0,o.kt)("p",null,"Note that:"),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},"This is non-standard interpolation support that is unique to the defaults list"),(0,o.kt)("li",{parentName:"ul"},"interpolation keys in the defaults list can not access values from the composed config because it does not yet\nexist when Hydra is processing the defaults list")),(0,o.kt)("p",null,"The Defaults List is described ",(0,o.kt)("a",{parentName:"p",href:"/docs/next/advanced/defaults_list"},"here"),"."),(0,o.kt)("div",{className:"admonition admonition-warning alert alert--danger"},(0,o.kt)("div",{parentName:"div",className:"admonition-heading"},(0,o.kt)("h5",{parentName:"div"},(0,o.kt)("span",{parentName:"h5",className:"admonition-icon"},(0,o.kt)("svg",{parentName:"span",xmlns:"http://www.w3.org/2000/svg",width:"12",height:"16",viewBox:"0 0 12 16"},(0,o.kt)("path",{parentName:"svg",fillRule:"evenodd",d:"M5.05.31c.81 2.17.41 3.38-.52 4.31C3.55 5.67 1.98 6.45.9 7.98c-1.45 2.05-1.7 6.53 3.53 7.7-2.2-1.16-2.67-4.52-.3-6.61-.61 2.03.53 3.33 1.94 2.86 1.39-.47 2.3.53 2.27 1.67-.02.78-.31 1.44-1.13 1.81 3.42-.59 4.78-3.42 4.78-5.56 0-2.84-2.53-3.22-1.25-5.61-1.52.13-2.03 1.13-1.89 2.75.09 1.08-1.02 1.8-1.86 1.33-.67-.41-.66-1.19-.06-1.78C8.18 5.31 8.68 2.45 5.05.32L5.03.3l.02.01z"}))),"warning")),(0,o.kt)("div",{parentName:"div",className:"admonition-content"},(0,o.kt)("p",{parentName:"div"},"Support for the old style will be removed in Hydra 1.2."))))}c.isMDXComponent=!0}}]);