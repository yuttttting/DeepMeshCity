from Model.models import deepmeshcity, deepmeshcity_taxibj, deepmeshcity_TPE


def build_model(configs):
    networks_map = {
        'DeepMeshCity': deepmeshcity.DeepMeshCity,
        'DeepMeshCity_TaxiBJ': deepmeshcity_taxibj.DeepMeshCity,
        'DeepMeshCity_TPE': deepmeshcity_TPE.DeepMeshCity,
    }
    model_type = configs.model_type

    if model_type in networks_map:
        Network = networks_map[model_type]
        network = Network(configs).to(configs.device)
        return network
    else:
        raise ValueError('Name of network unknown %s' % model_type)
