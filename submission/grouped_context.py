"""Presentation only: group already selected, verbatim passages by provenance."""
from retrieval import topics


def group(source):
    path=source['source'].replace('\\','/')
    title=source.get('title',source['heading'])
    root=title.split(' / ')[0].strip() or source['heading']
    # Document identity, not similarity of body text, defines membership.
    # Reference sections retain their own provenance; never infer a product owner.
    product='20-products' in path.split('/') and len(topics(root))==1
    return (product,path),root


def format_context(original,report,max_chars):
    groups={}
    for source in report['included']:
        identity,label=group(source)
        if identity not in groups:groups[identity]=(label,[])
        groups[identity][1].append(source)
    blocks=[];included=[]
    # Stable order by first appearance within each class.
    identities=[k for k in groups if k[0]]+[k for k in groups if not k[0]]
    for number,identity in enumerate(identities,1):
        label,sources=groups[identity]
        kind='Продуктовый документ' if identity[0] else 'Справочный или иной документ'
        lines=[f'=== Блок {number}. {kind}: {label} ===']
        for source in sources:
            lines.append(source['source']+' / '+source.get('title',source['heading'])+'\n'+source['text'])
        lines.append(f'=== Конец блока {number} ===')
        blocks.append('\n\n'.join(lines));included.extend(sources)
    result='\n\n'.join(blocks)
    if len(result)>max_chars:
        # Labels must never evict evidence or exceed the original budget.
        return original,dict(report,grouping_applied=False,grouping_reason='label_budget')
    return result,dict(report,included=included,characters=len(result),
                       grouping_applied=True,document_groups=len(groups))
