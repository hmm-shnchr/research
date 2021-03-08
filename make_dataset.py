##Extract the data to be used as dataset(data type is double-dictionary).
##Data structure of the dataset is that axis1 is host-halo(ex:MW039, MW038), axis2 is sub-halo.
def make_dataset(m_list, p_kind, h_param, param, ext_data, acc_sf):
    dataset = {}
    for m_key in m_list:
        dataset[m_key] = []
        for idx, parameter in enumerate(param[p_kind][m_key]):
            ##If the data used is after accretion(ext_data == After_Acc), extract after accretion point.
            ##The accretion point is taken from the acc_sf as start_i.
            start_i = 0
            if ext_data in ["After_acc", "All_acc"]:
                if acc_sf[m_key][idx] == -1:
                    ##The acc_sf == -1 is not accretion so skip this halo.
                    continue
                elif ext_data == "After_acc":
                    start_i = acc_sf[m_key][idx]
            if p_kind in ["x", "y", "z"]:
                ##If p_kind is coordinate, use relative value of sub-halo to host-halo.
                data = parameter[start_i:]
                host_i = data.size
                data -= h_param[p_kind][m_key][-host_i:]
                dataset[m_key].append(data)
            else:
                data = parameter[start_i:]
    return dataset
