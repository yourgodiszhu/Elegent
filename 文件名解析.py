import json
import os
import re



# def parse_file_name(file_name):
#     parts = file_name.rsplit(' ', 1)  # 按最后一个空格分
#     if len(parts) != 2:
#         raise ValueError("文件名格式不符合要求")
#
#     before_last_space, after_last_space = parts
#     key = file_name.split(' ', 1)[0]  # 第一个空格前第一个字符
#
#     # 最后一个空格后的颜色顺序
#     colors = after_last_space.split('+')
#     # 最后一个空格前的值顺序
#     values = before_last_space.split(' ')[-1].split('+')
#
#     if len(colors) != len(values):
#         raise ValueError("颜色数量和数值数量不匹配")
#
#     result = {key: {}}
#     for color, value in zip(colors, values):
#         if color == '红':
#             result[key]['SPRED'.lower() ]= value
#         elif color == '绿':
#             result[key]['SPGREEN'.lower() ] = value
#         else:
#             result[key][f'SP{color}'] = value  # 支持其他颜色
#
#     return result
from pypinyin import lazy_pinyin

def to_pinyin(text: str) -> str:
    # 如果是中文，就转拼音；如果不是，就保持原样
    if any('\u4e00' <= ch <= '\u9fff' for ch in text):
        return ''.join(lazy_pinyin(text))
    return text

def parse_file_name(file_name):
    # 按最后一个空格拆分
    file_name=str(file_name)
    parts = file_name.rsplit(' ', 1)
    # if len(parts) != 2:
    #     raise ValueError("文件名格式不符合要求")
    before_last_space, after_last_space = parts
    print(after_last_space)
    # 获取第一个空格前的字符串和第几轮
    first_space_split = file_name.split(' ',1)  # 拆分前两个空格
    # print('first_space_split',first_space_split)
    if len(first_space_split) < 2:
        raise ValueError("文件名格式不符合要求")
    first_part = first_space_split[0]  # OBTT
    first_part = to_pinyin(first_part)  # 转拼音
    second_part = first_space_split[1]  # 第1轮
    second_part=second_part.split(' ',1)[0]
    round_num=second_part[1]
    mapping = str.maketrans({
        "I": "1",
        "O": "0",
        "一": "1",
        "二": "2",
        "三": "3",
        "四": "4",
        "五": "5",
        "六": "6",
        "七": "7",
        "八": "8",
        "九": "9",
        "零": "0"
    })
    round_num = round_num.translate(mapping)
    print(round_num)
    # 提取轮次数字
    # round_num_match = re.search(r'\d+', second_part)
    # if not round_num_match:
    #     raise ValueError("未找到轮次数字")
    # round_num = round_num_match.group()

    # 构建键：OBTT_1
    key = f"{first_part}_{round_num}"
    if '+' not in after_last_space:
        colors = []
        values = []
        value, color = after_last_space.rsplit('-',1)
        colors.append(color)
        values.append(value)
        result = {key: {}}
        for color, value in zip(colors, values):
            if color == '红':
                result[key]['sporange'] = value
            elif color == '绿':
                result[key]['spgreen'] = value
            elif color == '黄':
                result[key]['cy5'] = value
    else:
        # 解析最后空格后的颜色顺序
        v1,v2 = after_last_space.split('+')
        # 解析最后空格前的值顺序
        colors = []
        values = []
        value,color = v1.rsplit('-',1)
        value1,color1 = v2.rsplit('-',1)
        colors.append(color)
        colors.append(color1)
        values.append(value)
        values.append(value1)
        # for zz in values:
        #     if '' in zz:
        #         values.remove(zz)
        # print('空格分开',before_last_space.split(' '))
        # print('before_last_space',before_last_space)
        # print('value',values)
        # values = values[-1].split('+')
        # print('value2',values)
        # if len(colors) != len(values):
        #     raise ValueError("颜色数量和数值数量不匹配")

        # 构建字典
        result = {key: {}}
        for color, value in zip(colors, values):
            if color == '红':
                result[key]['sporange'] = value
            elif color == '绿':
                result[key]['spgreen'] = value
            elif color == '黄':
                result[key]['cy5'] = value
        # else:
        #     result[key][f'sp{color}'] = value  # 支持其他颜色

    return result

for all_file in os.listdir(rf'F:\9_29\xxx\zzz\11'):
    folder_path = rf'F:\9_29\xxx\zzz\11\{all_file}'
# for file_name in os.listdir(folder_path):
#     print(file_name)
#     dic = parse_file_name(file_name)
#     full_path = os.path.join(folder_path, file_name)
#     for k in dic:
#         new_filename=k+'.mrxs'
#         new_dirname=k
#         new_full_path = os.path.join(folder_path, new_filename)
#         new_dirname = os.path.join(folder_path, new_dirname)
#         if os.path.isfile(full_path):
#             os.rename(full_path, new_full_path)
#         if os.path.isdir(full_path):
#             os.rename(full_path, new_dirname)
# txt_name = folder_path.split(os.sep)[-1].split(' ')[0] + '.txt'
#
# txt_path = os.path.join(folder_path, txt_name)
#
# with open(txt_path, 'w', encoding='utf-8') as f:
#     f.write(json.dumps(total_dict, ensure_ascii=False, indent=4))


    total_dict = {}

    # 获取 folder_path 第一个空格前的字符作为记事本文件名
    txt_name = folder_path.split(os.sep)[-1].split(' ')[0] + '.txt'
    txt_path = os.path.join(folder_path, txt_name)
    for file_name in os.listdir(folder_path):
        if not os.path.isdir(os.path.join(folder_path, file_name)):
            continue
        full_path = os.path.join(folder_path, file_name)
        if file_name.endswith('.mrxs'):
            continue
        print(f"处理: {file_name}")

        # 解析文件名
        try:
            dic = parse_file_name(file_name)
            print(dic)
        except Exception as e:
            print(f"{file_name} 解析失败: {e}")
            continue
        print(dic)
        # 合并到总字典

        # 重命名文件/文件夹
        for k in dic:
            new_filename = k + '.mrxs'
            new_dirname = k
            new_full_path = os.path.join(folder_path, new_filename)
            new_dir_path = os.path.join(folder_path, new_dirname)
            total_dict[k]=dic[k]

            if os.path.exists(full_path+'.mrxs'):
                print('full_path', full_path)
                print('new_full_path', new_full_path)
                os.rename(full_path+'.mrxs', new_full_path)
            # elif os.path.isdir(full_path):
            os.rename(full_path, new_dir_path)
    print(total_dict)
    # 写入记事本（JSON 格式，方便读取）
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(str(total_dict))

    print(f"完成，字典已写入 {txt_path}")