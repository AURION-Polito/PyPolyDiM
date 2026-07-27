# _LICENSE_HEADER_
#
# Copyright (C) 2026.
# Terms register on the GPL-3.0 license.
#
# This file can be redistributed and/or modified under the license terms.
#
# See top level LICENSE file for more details.
#
# This file can be used citing references in CITATION.cff file.

import argparse
import os.path
from pathlib import Path
from Brinkman_DF_PCC_2D.program_utilities import create_test, create_mesh, export_errors
from Brinkman_DF_PCC_2D.assembler import *
from pypolydim.export_vtk_utilities import ExportVTKUtilities
from pypolydim.assembler_utilities import assembler_utilities
import cProfile
from typing import List


def main():

    program_folder = str(Path(__file__).resolve().parent)

    parser = argparse.ArgumentParser()
    parser.add_argument('-order', '--method-order', dest='method_order', default=2, type=int, help="Method order (Default: 2)")
    parser.add_argument('-method', '--method-type', dest='method_type', default=1, type=int,
                        help="Method type: 0 - taylor_hood; 1 - vem_df_pcc_full; 2 - vem_df_pcc_reduced (Default: 1)")
    parser.add_argument('-test', '--test-id', dest='test_id', default=1, type=int,
                        help="Test type: 1 - StokesSinSin; 2 - Brinkman (Default: 1)")
    parser.add_argument('-mesh', '--mesh-type', dest='mesh_type', default=0, type=int,
                        help="Mesh type: 0 - Triangular; 1 - Minimal; 2 - Polygonal; 5 - Square (Default: 0)")
    parser.add_argument('-tol1', '--tolerance-1-d', dest='tolerance1_d', default=1.0e-12, type=float, help="Geometric Tolerance 1D (Default: 1.0e-12)")
    parser.add_argument('-tol2', '--tolerance-2-d', dest='tolerance2_d', default=1.0e-14, type=float, help="Geometric Tolerance 2D (Default: 1.0e-14)")
    parser.add_argument('-area', '--mesh-max-relative-area', dest='max_relative_area', default=0.1, type=float, help="Mesh max relative area (Default: 0.1)")
    parser.add_argument('-export', '--export-path', dest='export_path', default=program_folder + './Export/Brinkman_DF_PCC_2D', type=str, help="Export Path (Default: ./Export/Brinkman_DF_PCC_2D)")
    parser.add_argument('-import', '--import-path', dest='import_path', default='./', type=str, help="Mesh Import Path (Default: ./)")
    args = parser.parse_args()

    pr = cProfile.Profile()
    pr.enable()

    export_file_path = args.export_path
    if not os.path.exists(export_file_path):
        os.makedirs(export_file_path)

    # Mesh file path
    export_mesh_path = args.export_path + "/Mesh"
    if not os.path.exists(export_mesh_path):
        os.makedirs(export_mesh_path)

    mesh_type = polydim.pde_tools.mesh.pde_mesh_utilities.MeshGenerator_Types_2D(args.mesh_type)
    method_type = polydim.pde_tools.local_space_df_pcc_2_d.MethodTypes(args.method_type)
    method_order = args.method_order

    geometry_utilities_config = gedim.GeometryUtilitiesConfig()
    geometry_utilities_config.tolerance1_d = args.tolerance1_d
    geometry_utilities_config.tolerance2_d = args.tolerance2_d
    geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)
    mesh_utilities = gedim.MeshUtilities()
    vtk_utilities = ExportVTKUtilities()

    print("Set problem...")
    test = create_test(args.test_id)
    pde_domain = test.domain()
    boundary_info = test.boundary_info()

    print("Create mesh...")
    mesh_data = gedim.MeshMatrices()
    mesh = gedim.MeshMatricesDAO(mesh_data)
    create_mesh(geometry_utilities, mesh_utilities, mesh_type, args.max_relative_area, args.import_path, pde_domain, mesh)

    print("Export Mesh...")
    vtk_utilities.export_mesh(export_mesh_path, mesh)
    print('\x1b[35m' + "Mesh exported in: " + export_mesh_path + '\x1b[0m')

    print("Compute Geometric Properties...")
    mesh_geometric_data = polydim.pde_tools.mesh.pde_mesh_utilities.compute_mesh_2_d_geometry_data(geometry_utilities, mesh_utilities, mesh)

    print("Create Discrete Local Space...")
    reference_element_data = polydim.pde_tools.local_space_df_pcc_2_d.create_reference_element(method_type, method_order)
    mesh_connectivity_data = polydim.pde_tools.mesh.MeshMatricesDAO_mesh_connectivity_data(mesh)

    dof_manager = polydim.pde_tools.do_fs.DOFsManager()
    mesh_do_fs_info = polydim.pde_tools.local_space_df_pcc_2_d.set_mesh_do_fs_info(reference_element_data, mesh, boundary_info)
    num_mesh_do_fs_info = len(mesh_do_fs_info)
    do_fs_data: List[polydim.pde_tools.do_fs.DOFsManager.DOFsData] = [polydim.pde_tools.do_fs.DOFsManager.DOFsData() for _ in range(num_mesh_do_fs_info)]
    do_fs_data_indices: List[polydim.pde_tools.do_fs.DOFsManager.CellsDOFsIndicesData] = [polydim.pde_tools.do_fs.DOFsManager.CellsDOFsIndicesData() for _ in range(num_mesh_do_fs_info)]
    for n in range(num_mesh_do_fs_info):
        do_fs_data[n] = dof_manager.create_do_fs_2_d(mesh_do_fs_info[n], mesh_connectivity_data)
        do_fs_data_indices[n] = dof_manager.compute_cells_do_fs_indices(do_fs_data[n], dim=2)

    assembler_utilities_obj = assembler_utilities()
    count_do_fs_data = assembler_utilities_obj.count_do_fs(do_fs_data)

    if count_do_fs_data.num_total_boundary_do_fs == 0:
        count_do_fs_data.num_total_do_fs += 1 # lagrange

    print('\x1b[6;30;42m' + "Created discrete space with ", count_do_fs_data.num_total_do_fs, " DOFs and ",
          count_do_fs_data.num_total_strongs, " STRONG" + '\x1b[0m')

    print("Assemble...")
    assembler_data = assemble(geometry_utilities_config,
                              mesh,
                              mesh_geometric_data,
                              mesh_do_fs_info,
                              do_fs_data,
                              do_fs_data_indices,
                              count_do_fs_data,
                              reference_element_data,
                              test)

    print("Solve...")
    solve(count_do_fs_data, assembler_data)

    print("Compute Errors...")
    post_process_data = post_process_solution(geometry_utilities_config,
                                              mesh,
                                              mesh_geometric_data,
                                              do_fs_data,
                                              do_fs_data_indices,
                                              count_do_fs_data,
                                              reference_element_data,
                                              assembler_data,
                                              test)

    print("Export Solution and Errors...")

    export_errors(export_file_path, args.test_id, args.mesh_type, args.method_type, args.method_order, mesh, count_do_fs_data, post_process_data)

    file_name = export_file_path + '/VelocityX_' + str(args.test_id) + '_' + str(args.method_type) + '_' + str(method_order)
    vtk_utilities.export_solution_2(file_name, mesh, post_process_data.cell0_ds_numeric_velocity[0],
                                    cell0_d_exact_solution=post_process_data.cell0_ds_exact_velocity[0],
                                    cell2_ds_error_l2=post_process_data.cell2_ds_error_l2_pressure,
                                    cell2_ds_error_h1=post_process_data.cell2_ds_error_h1_velocity)
    print('\x1b[35m' + "Velocity X exported in: " + file_name + ".vtu" + '\x1b[0m')

    file_name = export_file_path + '/VelocityY_' + str(args.test_id) + '_' + str(args.method_type) + '_' + str(method_order)
    vtk_utilities.export_solution_2(file_name, mesh, post_process_data.cell0_ds_numeric_velocity[1],
                                    cell0_d_exact_solution=post_process_data.cell0_ds_exact_velocity[1],
                                    cell2_ds_error_l2=post_process_data.cell2_ds_error_l2_pressure,
                                    cell2_ds_error_h1=post_process_data.cell2_ds_error_h1_velocity)
    print('\x1b[35m' + "Velocity Y exported in: " + file_name + ".vtu" + '\x1b[0m')

    print('\x1b[6;30;42m' + "Finish" + '\x1b[0m')

    pr.disable()
    pr.dump_stats(export_file_path + "/program.prof")

if __name__=='__main__':

    main()
