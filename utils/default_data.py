budjet_total_budget_preset={
    "total_budget": 1000
}

budjet_items_preset = [
            {'item_1_id': 'item_1', 'item_1_return_coeff': '3.1', 'item_1_min_alloc': '0', 'item_1_max_alloc': '200'},
            {'item_2_id': 'item_2', 'item_2_return_coeff': '2.1', 'item_2_min_alloc': '0', 'item_2_max_alloc': '300'},
            {'item_3_id': 'item_3', 'item_3_return_coeff': '1.1', 'item_3_min_alloc': '0', 'item_3_max_alloc': '1000'}
]

global_constraints= {
    "total_budget": 100000.0,
    "total_power_kva": 50.0,
    "total_space_sqm": 10.0
}

datacenter_servers_preset = [
            {'id': 'SrvA', 'cost': '500', 'cpu_cores': '48', 'ram_gb': '256', 'storage_tb': '10', 'power_kva': '0.5',
             'space_sqm': '0.2'},
            {'id': 'SrvB', 'cost': '300', 'cpu_cores': '32', 'ram_gb': '128', 'storage_tb': '5', 'power_kva': '0.3',
             'space_sqm': '0.1'},
            {'id': 'SrvC', 'cost': '800', 'cpu_cores': '128', 'ram_gb': '512', 'storage_tb': '20', 'power_kva': '0.8',
             'space_sqm': '0.3'}
]

datacenter_services_preset = [
            {'id': 'WebPool', 'revenue_per_unit': '100', 'req_cpu_cores': '4', 'req_ram_gb': '8',
             'req_storage_tb': '0.1', 'max_units': '50'},
            {'id': 'DBFarm', 'revenue_per_unit': '200', 'req_cpu_cores': '8', 'req_ram_gb': '16',
             'req_storage_tb': '0.5', 'max_units': '20'},
            {'id': 'BatchProc', 'revenue_per_unit': '150', 'req_cpu_cores': '16', 'req_ram_gb': '32',
             'req_storage_tb': '0.2', 'max_units': '30'}
]

depot_location = {"id": "D1","x": 79.0,"y": 73.6}
customer_locations = [
        {"id": "C1","x": 100.0,"y": 254.6},
        {"id": "C2","x": 363.0,"y": 260.6},
        {"id": "C3","x": 426.0,"y": 93.6}
    ]
num_vehicles= 1
num_depots=1