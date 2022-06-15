import pandas as pd

delete_i = [82, 97, 98, 106, 110, 124, 149, 178, 181, 189, 211, 227, 229, 254]
delete_ii = [13, 34, 35, 48, 60, 88, 96, 125, 130]
delete_iii = [0, 3, 27, 36, 39, 43, 50, 56, 65, 66, 71, 76, 81, 87, 88, 89, 91, 93, 96, 101, 102, 109, 119, 123,
              127, 136, 139, 165, 178]
delete_iv = [20, 26, 76, 94, 120, 131, 137, 152, 269]
delete_v = [5, 21, 25, 47, 51, 120, 124, 125, 130, 153, 157, 183, 188, 192]
delete_vi = [17, 29, 30, 35, 70, 76, 81, 98, 102, 109, 110, 151, 156, 166, 176]
delete_vii = [0, 4, 6, 16, 36, 39, 47, 54, 56, 63, 68, 73, 99, 154, 170, 187]
delete_viii = [0, 8, 26, 31, 35, 37, 39, 43, 52, 58, 71, 82, 83, 104, 109, 141, 149, 161, 163, 174]
delete_ix = [0, 18, 27, 37, 49, 56, 61, 72, 78, 82, 94, 99, 111, 116, 156, 163, 164, 181, 183, 184, 193, 194, 196, 232,
             233]
delete_x = [19, 38, 46, 59, 67, 83, 94, 100, 111, 112, 160, 162, 174]

delete_df = pd.DataFrame(data={'i': [delete_i], 'ii': [delete_ii], 'iii': [delete_iii], 'iv': [delete_iv],
                               'v': [delete_v], 'vi': [delete_vi], 'vii': [delete_vii], 'viii': [delete_viii],
                               'ix': [delete_ix], 'x': [delete_x]})
delete_df.to_csv("delete.csv", index=False)

move_to_i = ['ii_94', 'iii_75', 'iii_132', 'iv_273', 'v_161', 'ix_112']
move_to_ii = ['i_17', 'i_75', 'i_142', 'i_157', 'i_223', 'i_224', 'i_257', 'iii_37', 'iii_73', 'iii_82', 'iii_83',
              'iii_86', 'iii_162', 'iii_183', 'iv_40', 'iv_162', 'iv_164', 'iv_258', 'v_67', 'v_68', 'v_180', 'vi_121',
              'vi_124', 'viii_127', 'viii_140', 'ix_52', 'x_47']
move_to_iii = ['i_39', 'i_165', 'ii_20', 'iv_38', 'iv_72', 'iv_122', 'iv_192', 'iv_221', 'vii_184', 'viii_4', 'viii_88',
               'viii_164', 'ix_136']
move_to_iv = ['iii_61', 'iii_98', 'v_162', 'vi_45', 'vi_64', 'vi_108', 'vi_136', 'viii_77', 'viii_148', 'ix_62',
              'ix_212']
move_to_v = ['i_248', 'ii_10', 'ii_77', 'iii_33', 'iv_47', 'iv_74', 'iv_238', 'vi_82', 'x_40', 'x_82', 'x_168']
move_to_vi = ['v_33', 'v_170', 'vii_148']
move_to_vii = []
move_to_viii = []
move_to_ix = ['iii_54', 'iii_131', 'iv_14', 'iv_150', 'x_70']
move_to_x = ['i_139', 'ii_43', 'ii_126', 'iv_17', 'v_30', 'v_60', 'v_69', 'v_101', 'v_104', 'v_110', 'vi_47', 'vi_100',
             'vi_145', 'ix_4', 'ix_44', 'ix_197']

move_df = pd.DataFrame(data={'i': [move_to_i], 'ii': [move_to_ii], 'iii': [move_to_iii], 'iv': [move_to_iv],
                             'v': [move_to_v], 'vi': [move_to_vi], 'vii': [move_to_vii], 'viii': [move_to_viii],
                             'ix': [move_to_ix], 'x': [move_to_x]})
move_df.to_csv("move.csv", index=False)
