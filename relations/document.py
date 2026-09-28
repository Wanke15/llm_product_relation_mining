from __future__ import annotations

from .core import digest, model_product

# 影响关系判定的字段集 = 实际发送给判定的完整商品模型（不含图片 URL）。
# 只要任一判定输入字段变化，product_hash 就变化，相关缓存随之失效。


def product_hash(p) -> str:
    return digest(model_product(p))


# embedding 输入模板版本；改动模板即改变 embedding_text_hash，旧向量缓存自动失效。
TEMPLATE_VERSION = 'embedding-text-v1'


def _join(items):
    return '、'.join(str(x) for x in items if x not in (None, ''))


def embedding_text(p) -> str:
    """确定性商品文档，供 embedding 使用。

    只含语义字段：名称 / 部门·类目组·类目·细分类目 / 特征·风格·有效主题 / 属性·可选颜色·季节。
    省略空值与内部无意义编码；价格与图片 URL 不进入文本。
    available_colors 是 SPU 可选颜色集合，不是单件配色。
    """
    lines = []
    if p.get('name'):
        lines.append('商品名：' + p['name'])
    cat = _join([p.get('branch'), p.get('category_group'), p.get('category'), p.get('subcategory')])
    if cat:
        lines.append('类目：' + cat)
    if p.get('features'):
        lines.append('特征：' + p['features'])
    if p.get('style'):
        lines.append('风格：' + p['style'])
    if p.get('theme'):
        lines.append('主题：' + p['theme'])
    if p.get('attribute_text'):
        lines.append('属性：' + p['attribute_text'])
    if p.get('available_colors'):
        lines.append('可选颜色：' + _join(p['available_colors']))
    if p.get('season'):
        lines.append('季节：' + p['season'])
    return '。'.join(lines)


def embedding_text_hash(p) -> str:
    return digest({'template': TEMPLATE_VERSION, 'text': embedding_text(p)})