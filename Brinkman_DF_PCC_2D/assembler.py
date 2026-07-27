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

import numpy as np
from pypolydim import polydim, gedim
from Brinkman_DF_PCC_2D.test_definition import ITest
import scipy.sparse.linalg as sla
from pypolydim.assembler_utilities import assembler_utilities
from scipy.sparse import coo_array
from typing import List


class ProblemData:

    global_matrix_a: coo_array
    dirichlet_matrix_a: coo_array

    def __init__(self):
        self.global_matrix_a_data = assembler_utilities.SparseMatrix()
        self.dirichlet_matrix_a_data = assembler_utilities.SparseMatrix()
        self.right_hand_side: np.ndarray = np.ndarray(0)
        self.solution: np.ndarray = np.ndarray(0)
        self.solution_dirichlet: np.ndarray = np.ndarray(0)


class PostProcessData:

    def __init__(self):
        self.cell0_ds_numeric_velocity: List[np.ndarray] = [np.ndarray(0) for _ in range(2)]
        self.cell0_ds_exact_velocity: List[np.ndarray] = [np.ndarray(0) for _ in range(2)]

        self.cell2_ds_error_l2_pressure: np.ndarray = np.ndarray(0)
        self.cell2_ds_norm_l2_pressure: np.ndarray = np.ndarray(0)
        self.error_l2_pressure: float = 0.0
        self.norm_l2_pressure: float = 0.0

        self.cell2_ds_error_h1_velocity: np.ndarray = np.ndarray(0)
        self.cell2_ds_norm_h1_velocity: np.ndarray = np.ndarray(0)
        self.error_h1_velocity: float = 0.0
        self.norm_h1_velocity: float = 0.0

        self.mesh_size: float = np.inf
        self.residual_norm: float = np.inf


def compute_strong_term(cell2_d_index: int,
                        mesh: gedim.MeshMatricesDAO,
                        mesh_do_fs_info: List[polydim.pde_tools.do_fs.DOFsManager.MeshDOFsInfo],
                        do_fs_data: List[polydim.pde_tools.do_fs.DOFsManager.DOFsData],
                        count_do_fs_data: assembler_utilities.CountDOFsData,
                        reference_element_data: polydim.pde_tools.local_space_df_pcc_2_d.ReferenceElement_Data,
                        local_space_data: polydim.pde_tools.local_space_df_pcc_2_d.LocalSpace_Data,
                        test: ITest,
                        assembler_data: ProblemData) -> None:

    for h in range(2):
        # Assemble strong boundary condition on Cell0Ds
        for v in range(mesh.cell2_d_number_vertices(cell2_d_index)):

            cell0_d_index = mesh.cell2_d_vertex(cell2_d_index, v)
            boundary_info = mesh_do_fs_info[h].cells_boundary_info[0][cell0_d_index]

            if boundary_info.type != polydim.pde_tools.do_fs.DOFsManager.MeshDOFsInfo.BoundaryInfo.BoundaryTypes.strong:
                continue

            coordinates = np.expand_dims(mesh.cell0_d_coordinates(cell0_d_index), axis=1)

            strong_boundary_values = test.strong_boundary_condition(boundary_info.marker, coordinates)[h]

            local_do_fs = do_fs_data[h].cells_do_fs[0][cell0_d_index]

            assert len(local_do_fs) == len(strong_boundary_values)

            for loc_i in range(len(local_do_fs)):

                local_dof_i = local_do_fs[loc_i]

                match local_dof_i.type:
                    case polydim.pde_tools.do_fs.DOFsManager.DOFsData.DOF.Types.strong:
                        assembler_data.solution_dirichlet[local_dof_i.global_index + count_do_fs_data.offset_strongs[h]] = strong_boundary_values[loc_i]
                        pass
                    case polydim.pde_tools.do_fs.DOFsManager.DOFsData.DOF.Types.dof:
                        pass
                    case _:
                        raise ValueError("Unknown DOF Type")

        # Assemble strong boundary condition on Cell1Ds
        for ed in range(mesh.cell2_d_number_edges(cell2_d_index)):

            cell1_d_index = mesh.cell2_d_edge(cell2_d_index, ed)

            boundary_info = mesh_do_fs_info[h].cells_boundary_info[1][cell1_d_index]
            local_do_fs = do_fs_data[h].cells_do_fs[1][cell1_d_index]

            if (boundary_info.type != polydim.pde_tools.do_fs.DOFsManager.MeshDOFsInfo.BoundaryInfo.BoundaryTypes.strong
                    or len(local_do_fs) == 0):
                continue

            edge_do_fs_coordinates = polydim.pde_tools.local_space_df_pcc_2_d.velocity_edge_dofs_coordinates(reference_element_data, local_space_data, ed)

            strong_boundary_values = test.strong_boundary_condition(boundary_info.marker, edge_do_fs_coordinates)[h]

            assert len(local_do_fs) == len(strong_boundary_values)

            for loc_i in range(len(local_do_fs)):

                local_dof_i = local_do_fs[loc_i]

                match local_dof_i.type:
                    case polydim.pde_tools.do_fs.DOFsManager.DOFsData.DOF.Types.strong:
                        assembler_data.solution_dirichlet[local_dof_i.global_index + count_do_fs_data.offset_strongs[h]] = strong_boundary_values[loc_i]
                        pass
                    case polydim.pde_tools.do_fs.DOFsManager.DOFsData.DOF.Types.dof:
                        pass
                    case _:
                        raise ValueError("Unknown DOF Type")


def compute_weak_term(cell2_d_index: int,
                      mesh: gedim.MeshMatricesDAO,
                      mesh_geometric_data: gedim.MeshUtilities.MeshGeometricData2D,
                      mesh_do_fs_info: List[polydim.pde_tools.do_fs.DOFsManager.MeshDOFsInfo],
                      do_fs_data: List[polydim.pde_tools.do_fs.DOFsManager.DOFsData],
                      count_do_fs_data: assembler_utilities.CountDOFsData,
                      reference_element_data: polydim.pde_tools.local_space_df_pcc_2_d.ReferenceElement_Data,
                      local_space_data: polydim.pde_tools.local_space_df_pcc_2_d.LocalSpace_Data,
                      test: ITest,
                      assembler_data: ProblemData) -> None:

    for h in range(2):
        num_vertices = mesh.cell2_d_number_vertices(cell2_d_index)

        for ed in range(num_vertices):
            cell1_d_index = mesh.cell2_d_edge(cell2_d_index, ed)

            boundary_info = mesh_do_fs_info[h].cells_boundary_info[1][cell1_d_index]

            if boundary_info.type != polydim.pde_tools.do_fs.DOFsManager.MeshDOFsInfo.BoundaryInfo.BoundaryTypes.weak:
                continue

            # compute vem values
            weak_reference_segment = gedim.quadrature.Quadrature_Gauss1D.fill_points_and_weights(2 * reference_element_data.order)
            points_curvilinear_coordinates = weak_reference_segment.points[0, :]

            # map edge internal quadrature points
            edge_start = mesh_geometric_data.cell2_ds_vertices[cell2_d_index][:, ed] \
                if mesh_geometric_data.cell2_ds_edge_directions[cell2_d_index][ed] \
                else mesh_geometric_data.cell2_ds_vertices[cell2_d_index][:, (ed + 1) % num_vertices]
            edge_tangent = mesh_geometric_data.cell2_ds_edge_tangents[cell2_d_index][:, ed]
            direction = 1.0 if mesh_geometric_data.cell2_ds_edge_directions[cell2_d_index][ed] else -1.0
            num_edge_weak_quadrature_points = weak_reference_segment.points.shape[1]

            weak_quadrature_points = np.zeros([3, num_edge_weak_quadrature_points])
            for q in range(num_edge_weak_quadrature_points):
                weak_quadrature_points[:, q] = edge_start + direction * weak_reference_segment.points[0, q] * edge_tangent

            edge_length = mesh_geometric_data.cell2_ds_edge_lengths[cell2_d_index][ed]
            weak_quadrature_weights = weak_reference_segment.weights * edge_length

            neumann_values = test.weak_boundary_condition(boundary_info.marker, weak_quadrature_points)[h]
            weak_basis_function_values \
                = polydim.pde_tools.local_space_df_pcc_2_d.velocity_basis_functions_values_on_edge(ed,
                                                                                                   reference_element_data,
                                                                                                   local_space_data,
                                                                                                   points_curvilinear_coordinates)

            # compute values of Neumann
            neumann_contributions = -(weak_basis_function_values.T @
                                     np.diag(weak_quadrature_weights) @
                                     neumann_values)

            for p in range(2):
                cell0_d_index = mesh.cell1_d_vertex(cell1_d_index, p)
                local_do_fs = do_fs_data[h].cells_do_fs[0][cell0_d_index]

                for loc_i in range(len(local_do_fs)):
                    local_dof_i = local_do_fs[loc_i]

                    match local_dof_i.type:
                        case polydim.pde_tools.do_fs.DOFsManager.DOFsData.DOF.Types.strong:
                            continue
                        case polydim.pde_tools.do_fs.DOFsManager.DOFsData.DOF.Types.dof:
                            assembler_data.right_hand_side[local_dof_i.global_index + count_do_fs_data.offset_do_fs[h]] += neumann_contributions[p]
                            pass
                        case _:
                            raise ValueError("Unknown DOF Type")

            local_do_fs = do_fs_data[h].cells_do_fs[1][cell1_d_index]
            for loc_i in range(len(local_do_fs)):
                local_dof_i = local_do_fs[loc_i]

                match local_dof_i.type:
                    case polydim.pde_tools.do_fs.DOFsManager.DOFsData.DOF.Types.strong:
                        continue
                    case polydim.pde_tools.do_fs.DOFsManager.DOFsData.DOF.Types.dof:
                        assembler_data.right_hand_side[local_dof_i.global_index + count_do_fs_data.offset_do_fs[h]] += neumann_contributions[loc_i + 2]
                        pass
                    case _:
                        raise ValueError("Unknown DOF Type")


def solve(count_do_fs_data: assembler_utilities.CountDOFsData,
          assembler_data: ProblemData) -> None:

    if count_do_fs_data.num_total_strongs > 0:
        assembler_data.right_hand_side -= assembler_data.dirichlet_matrix_a @ assembler_data.solution_dirichlet

    assembler_data.solution = sla.spsolve(assembler_data.global_matrix_a.tocsc(), assembler_data.right_hand_side)


def assemble(geometry_utilities_config: gedim.GeometryUtilitiesConfig,
             mesh: gedim.MeshMatricesDAO,
             mesh_geometric_data: gedim.MeshUtilities.MeshGeometricData2D,
             mesh_do_fs_info: List[polydim.pde_tools.do_fs.DOFsManager.MeshDOFsInfo],
             do_fs_data: List[polydim.pde_tools.do_fs.DOFsManager.DOFsData],
             do_fs_data_indices: List[polydim.pde_tools.do_fs.DOFsManager.CellsDOFsIndicesData],
             count_do_fs_data: assembler_utilities.CountDOFsData,
             reference_element_data: polydim.pde_tools.local_space_df_pcc_2_d.ReferenceElement_Data,
             test: ITest) -> ProblemData:

    result = ProblemData()

    result.right_hand_side = np.zeros([count_do_fs_data.num_total_do_fs])
    result.solution_dirichlet = np.zeros([count_do_fs_data.num_total_strongs])

    equation = polydim.pde_tools.equations.EllipticEquation()
    assembler_utils_obj = assembler_utilities()

    for c in range(mesh.cell2_d_total_number()):

        local_space_data = polydim.pde_tools.local_space_df_pcc_2_d.create_local_space(geometry_utilities_config.tolerance1_d,
                                            geometry_utilities_config.tolerance2_d,
                                            mesh_geometric_data,
                                            c,
                                            reference_element_data)

        velocity_basis_functions_values = polydim.pde_tools.local_space_df_pcc_2_d.velocity_basis_functions_values(reference_element_data, local_space_data, polydim.vem.df_pcc.ProjectionTypes.pi0k)
        velocity_basis_functions_derivatives_values = polydim.pde_tools.local_space_df_pcc_2_d.velocity_basis_functions_derivative_values(reference_element_data, local_space_data, polydim.vem.df_pcc.ProjectionTypes.pi_nabla)
        velocity_basis_functions_divergence_values = polydim.pde_tools.local_space_df_pcc_2_d.velocity_basis_functions_divergence_values(reference_element_data, local_space_data)
        pressure_basis_functions_values = polydim.pde_tools.local_space_df_pcc_2_d.pressure_basis_functions_values(reference_element_data, local_space_data)

        cell2_d_internal_quadrature = polydim.pde_tools.local_space_df_pcc_2_d.internal_quadrature(reference_element_data, local_space_data)
        weights = cell2_d_internal_quadrature.weights

        inverse_diffusion_term_values = test.inverse_diffusion_term(cell2_d_internal_quadrature.points)
        fluid_viscosity_values = test.fluid_viscosity(cell2_d_internal_quadrature.points)
        source_term_values = test.source_term(cell2_d_internal_quadrature.points)
        divergence_term_values = test.divergence_term(cell2_d_internal_quadrature.points)

        local_a = equation.compute_cell_diffusion_matrix(fluid_viscosity_values,
                                                         velocity_basis_functions_derivatives_values,
                                                         weights)

        mu_max = float(np.max(abs(fluid_viscosity_values)))
        local_a_stab = mu_max * polydim.pde_tools.local_space_df_pcc_2_d.velocity_stabilization_matrix(reference_element_data, local_space_data, polydim.vem.df_pcc.ProjectionTypes.pi_nabla)
        local_a += local_a_stab

        local_a += equation.compute_cell_diffusion_matrix(inverse_diffusion_term_values,
                                                         velocity_basis_functions_values,
                                                         weights)

        k_max_component = [float(np.max(abs(inverse_diffusion_term_values[i]))) for i in range(9)]
        local_a_stab = max(k_max_component) * polydim.pde_tools.local_space_df_pcc_2_d.velocity_stabilization_matrix(reference_element_data, local_space_data, polydim.vem.df_pcc.ProjectionTypes.pi0k)
        local_a += local_a_stab

        local_b = pressure_basis_functions_values.T @ np.diag(weights) @ velocity_basis_functions_divergence_values

        local_rhs = equation.compute_cell_forcing_term(source_term_values,
                                                       velocity_basis_functions_values,
                                                       weights)

        local_div = equation.compute_cell_forcing_term(divergence_term_values,
                                                       pressure_basis_functions_values,
                                                       weights)

        local_count_do_fs = assembler_utils_obj.local_count_do_fs(2, c, do_fs_data)
        num_local_do_fs_pressure = pressure_basis_functions_values.shape[1]
        num_local_do_fs_velocity = velocity_basis_functions_values[0].shape[1]

        elemental_rhs = np.concatenate([local_rhs, local_div], axis=0)
        elemental_matrix = np.block([[local_a, -local_b.T], [local_b, np.zeros([num_local_do_fs_pressure, num_local_do_fs_pressure])]])

        assert local_count_do_fs.num_total_do_fs == num_local_do_fs_pressure + num_local_do_fs_velocity

        local_to_global_data = assembler_utilities.LocalMatrixToGlobalMatrixDOFsData(do_fs_data_indices, local_count_do_fs.offset_do_fs, count_do_fs_data.offset_do_fs, count_do_fs_data.offset_strongs)

        assembler_utilities.assemble_local_matrix_to_global_matrix(2,
                                                                   c,
                                                                   local_to_global_data,
                                                                   local_to_global_data,
                                                                   elemental_matrix,
                                                                   result.global_matrix_a_data,
                                                                   result.dirichlet_matrix_a_data,
                                                                   elemental_rhs,
                                                                   result.right_hand_side)

        if count_do_fs_data.num_total_boundary_do_fs == 0:

            mean_value_pressure = pressure_basis_functions_values.T @ cell2_d_internal_quadrature.weights

            h1 = 3
            num_global_offset_lagrange = int(count_do_fs_data.num_total_do_fs - 1)
            for loc_i in range(len(do_fs_data[h1].cells_global_do_fs[2][c])):
                global_dof_i = do_fs_data[h1].cells_global_do_fs[2][c][loc_i]
                local_dof: List[polydim.pde_tools.do_fs.DOFsManager.DOFsData.DOF] = do_fs_data[h1].cells_do_fs[global_dof_i.dimension][global_dof_i.cell_index]
                local_dof_i: polydim.pde_tools.do_fs.DOFsManager.DOFsData.DOF = local_dof[global_dof_i.dof_index]
                global_index_i = local_dof_i.global_index + count_do_fs_data.offset_do_fs[h1]

                result.global_matrix_a_data.row.append(global_index_i)
                result.global_matrix_a_data.col.append(num_global_offset_lagrange)
                result.global_matrix_a_data.data.append(float(mean_value_pressure[loc_i]))

                result.global_matrix_a_data.row.append(num_global_offset_lagrange)
                result.global_matrix_a_data.col.append(global_index_i)
                result.global_matrix_a_data.data.append(float(mean_value_pressure[loc_i]))


        compute_strong_term(c, mesh, mesh_do_fs_info, do_fs_data, count_do_fs_data, reference_element_data, local_space_data,
                            test, result)
        compute_weak_term(c, mesh, mesh_geometric_data, mesh_do_fs_info, do_fs_data, count_do_fs_data, reference_element_data,
                          local_space_data, test, result)

    result.global_matrix_a = result.global_matrix_a_data.create(count_do_fs_data.num_total_do_fs, count_do_fs_data.num_total_do_fs)
    result.dirichlet_matrix_a = result.dirichlet_matrix_a_data.create(count_do_fs_data.num_total_do_fs, count_do_fs_data.num_total_strongs)

    return result


def post_process_solution(geometry_utilities_config: gedim.GeometryUtilitiesConfig,
                          mesh: gedim.MeshMatricesDAO,
                          mesh_geometric_data: gedim.MeshUtilities.MeshGeometricData2D,
                          do_fs_data: List[polydim.pde_tools.do_fs.DOFsManager.DOFsData],
                          do_fs_data_indices: List[polydim.pde_tools.do_fs.DOFsManager.CellsDOFsIndicesData],
                          count_do_fs_data: assembler_utilities.CountDOFsData,
                          reference_element_data: polydim.pde_tools.local_space_df_pcc_2_d.ReferenceElement_Data,
                          assembler_data: ProblemData,
                          test: ITest) -> PostProcessData:

    result = PostProcessData()

    result.residual_norm = 0.0
    if count_do_fs_data.num_total_do_fs > 0:
        residual = assembler_data.global_matrix_a @ assembler_data.solution - assembler_data.right_hand_side
        result.residual_norm = float(np.linalg.norm(residual))

    for d in range(2):
        result.cell0_ds_numeric_velocity[d] = np.zeros([mesh.cell0_d_total_number()])
        result.cell0_ds_exact_velocity[d] = np.zeros([mesh.cell0_d_total_number()])

    for p in range(mesh.cell0_d_total_number()):

        velocity_values = test.exact_velocity(np.expand_dims(mesh.cell0_d_coordinates(p), axis=1))
        for d in range(2):
            result.cell0_ds_exact_velocity[d][p] = velocity_values[d][0]

        for d in range(2):
            local_do_fs = do_fs_data[d].cells_do_fs[0][p]

            for loc_i in range(len(local_do_fs)):

                local_dof_i = local_do_fs[loc_i]

                match local_dof_i.type:
                    case polydim.pde_tools.do_fs.DOFsManager.DOFsData.DOF.Types.strong:
                        result.cell0_ds_numeric_velocity[d][p] = assembler_data.solution_dirichlet[local_dof_i.global_index + count_do_fs_data.offset_strongs[d]]
                        pass
                    case polydim.pde_tools.do_fs.DOFsManager.DOFsData.DOF.Types.dof:
                        result.cell0_ds_numeric_velocity[d][p] = assembler_data.solution[local_dof_i.global_index + count_do_fs_data.offset_do_fs[d]]
                    case _:
                        raise ValueError("Unknown DOF Type")


    result.cell2_ds_error_l2_pressure = np.zeros([mesh.cell2_d_total_number()])
    result.cell2_ds_norm_l2_pressure  = np.zeros([mesh.cell2_d_total_number()])
    result.cell2_ds_error_h1_velocity = np.zeros([mesh.cell2_d_total_number()])
    result.cell2_ds_norm_h1_velocity = np.zeros([mesh.cell2_d_total_number()])
    result.mesh_size = 0.0

    assembler_utilities_obj = assembler_utilities()
    for c in range(mesh.cell2_d_total_number()):

        local_space_data = polydim.pde_tools.local_space_df_pcc_2_d.create_local_space(geometry_utilities_config.tolerance1_d,
                                            geometry_utilities_config.tolerance2_d,
                                            mesh_geometric_data,
                                            c,
                                            reference_element_data)

        cell2_d_internal_quadrature = polydim.pde_tools.local_space_df_pcc_2_d.internal_quadrature(reference_element_data, local_space_data)

        velocity_basis_functions_derivatives_values = polydim.pde_tools.local_space_df_pcc_2_d.velocity_basis_functions_derivative_values(reference_element_data,
                                                                                                                                          local_space_data,
                                                                                                                                          polydim.vem.df_pcc.ProjectionTypes.pi0km1_der)
        pressure_basis_functions_values = polydim.pde_tools.local_space_df_pcc_2_d.pressure_basis_functions_values(reference_element_data, local_space_data)

        exact_pressure_values = test.exact_pressure(cell2_d_internal_quadrature.points)
        exact_velocity_derivatives_values = test.exact_derivatives_velocity(cell2_d_internal_quadrature.points)

        local_count_do_fs = assembler_utilities_obj.local_count_do_fs(2, c, do_fs_data)
        do_fs_values = assembler_utilities_obj.global_solution_to_local_solution(2,
                                                                                 c,
                                                                                 do_fs_data_indices,
                                                                                 count_do_fs_data,
                                                                                 local_count_do_fs,
                                                                                 assembler_data.solution,
                                                                                 assembler_data.solution_dirichlet)

        num_dof_handlers = len(do_fs_data)
        do_fs_velocity_values = do_fs_values[0:local_count_do_fs.offset_do_fs[num_dof_handlers - 1]]
        do_fs_pressure_values = do_fs_values[local_count_do_fs.offset_do_fs[num_dof_handlers - 1]:]

        local_error_l2_pressure = (pressure_basis_functions_values @ do_fs_pressure_values - exact_pressure_values)**2
        local_norm_l2_pressure = (pressure_basis_functions_values @ do_fs_pressure_values)**2

        result.cell2_ds_error_l2_pressure[c] = np.sum(cell2_d_internal_quadrature.weights * local_error_l2_pressure)
        result.cell2_ds_norm_l2_pressure[c] = np.sum(cell2_d_internal_quadrature.weights * local_norm_l2_pressure)

        local_error_h1_velocity = np.zeros(cell2_d_internal_quadrature.points.shape[1])
        local_norm_h1_velocity = np.zeros(cell2_d_internal_quadrature.points.shape[1])
        for d1 in range(2):
            for d2 in range(2):
                local_error_h1_velocity += (velocity_basis_functions_derivatives_values[2 * d1 + d2] @ do_fs_velocity_values - exact_velocity_derivatives_values[3 * d1 + d2])**2
                local_norm_h1_velocity += (velocity_basis_functions_derivatives_values[2 * d1 + d2] @ do_fs_velocity_values)**2

        result.cell2_ds_error_h1_velocity [c] = np.sum(cell2_d_internal_quadrature.weights * local_error_h1_velocity )
        result.cell2_ds_norm_h1_velocity [c] = np.sum(cell2_d_internal_quadrature.weights * local_norm_h1_velocity )

        if mesh_geometric_data.cell2_ds_diameters[c] > result.mesh_size:
            result.mesh_size = mesh_geometric_data.cell2_ds_diameters[c]


    result.error_l2_pressure = np.sqrt(np.sum(result.cell2_ds_error_l2_pressure))
    result.norm_l2_pressure = np.sqrt(np.sum(result.cell2_ds_norm_l2_pressure))
    result.error_h1_velocity = np.sqrt(np.sum(result.cell2_ds_error_h1_velocity))
    result.norm_h1_velocity = np.sqrt(np.sum(result.cell2_ds_norm_h1_velocity))

    return result

