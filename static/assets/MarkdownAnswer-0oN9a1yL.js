import{j as o}from"./react-KFtmQ_AB.js";import{v as l,u as m,r as f,M as p,a as d,b as u,c as x}from"./markdown-BalBmNpq.js";const h=m().use(f);function k(r){const a=[];l(h.parse(r),e=>{if(e.type!=="code"&&e.type!=="inlineCode")return;const s=e.position?.start.offset,i=e.position?.end.offset;s!==void 0&&i!==void 0&&a.push([s,i])});const t=e=>e.replace(/\\\[([\s\S]*?)\\\]/g,(s,i)=>`
$$
${i}
$$
`).replace(/\\\(([\s\S]*?)\\\)/g,(s,i)=>`$${i}$`);let n=0,c="";for(const[e,s]of a)c+=t(r.slice(n,e))+r.slice(e,s),n=s;return c+t(r.slice(n))}function j({content:r}){return o.jsx("div",{className:"markdown-answer assistant-answer",children:o.jsx(p,{remarkPlugins:[u,x],rehypePlugins:[d],components:{a:({children:a,...t})=>o.jsx("a",{...t,target:"_blank",rel:"noreferrer",children:a}),code:({children:a,className:t,...n})=>o.jsx("code",{...n,className:t||"inline-code",children:a})},children:k(r)})})}export{j as MarkdownAnswer};
