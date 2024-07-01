import numpy as np
import meshio as io
import cheartio as chio
from scipy.spatial import KDTree, ConvexHull, distance_matrix
from subprocess import Popen

params = {'ncells': 80,
        'window': 10,
        'min_cell_distance': 40,
        'min_post_distance': 10,
        'min_bound_distance': 30,
        'cell_long_axis': 50,
        'cell_short_axis': 20}


def seed_random_cells(bk, bk_dist):
    maxit = 100000
    bksize = bk.shape[0] - params['min_post_distance']*2

    bk_cell_region = bk_dist > params['min_post_distance']
    bk_cell_region[:,0:params['min_post_distance']] = 0       # I don't want cells at the border either
    bk_cell_region[:,-params['min_post_distance']::] = 0


    xycells = np.zeros([params['ncells'], 2])
    for i in range(params['ncells']):
        loop = True
        it = 0
        while loop and it < maxit:
            xcell = np.random.rand()*bksize + params['min_post_distance']
            bkx = bk_cell_region[int(np.floor(xcell))]*bk_cell_region[int(np.ceil(xcell))]
            aux = np.arange(len(bkx))[bkx]
            ylims = [aux[0], aux[-1]]

            ycell = np.random.rand()*(ylims[1]-ylims[0]) + ylims[0]

            xycell = np.array([xcell, ycell])
            if i == 0:
                xycells[i] = xycell
                break

            dist = np.linalg.norm(xycells[:i] - xycell, axis = 1)
            if np.min(dist) > params['min_cell_distance']:
                xycells[i] = xycell
                loop = False
            it+=1

        if it == maxit:
            print('Sedding failed. Cells seeded: ' + str(i))
            xycells = xycells[0:i]
            break

    return xycells


def get_utc_images(bk):
    i = np.arange(bk.shape[0])
    j = np.arange(bk.shape[1])
    I, J = np.meshgrid(j, i)

    axis_dir = J/np.max(J)
    trans_dir = np.zeros_like(axis_dir)
    for i in range(J.shape[0]):
        lims = np.where(bk[i,:]>0)[0][np.array([0,-1])]
        size = lims[1]-lims[0]
        trans_dir[i] = (I[i]-lims[0])/size

    trans_dir[trans_dir < 0] = 0
    trans_dir[trans_dir > 1] = 1

    return axis_dir, trans_dir


def get_utc(ij, axis_dir, trans_dir):
    axis = axis_dir[ij[:,0], ij[:,1]]
    trans = trans_dir[ij[:,0], ij[:,1]]
    utc = np.vstack([axis, trans]).T

    return utc


def assign_elements_to_cells(xycells, xyz, ien, elongation_factor):
    # # Map bk_dist to mesh # TODO use this as y so the cells follow the boundary
    # ij_nodes = np.floor(mesh.points/pixel_size).astype(int)
    # mesh.point_data['bk_dist'] = bk_dist[ij_nodes[:,0], ij_nodes[:,1]]

    xyz_aux = np.copy(xyz)
    xyz_aux[:,1] *= elongation_factor
    midpoints = np.mean(xyz_aux[ien], axis=1)

    xycells_aux = np.copy(xycells)
    xycells_aux[:,1] *= elongation_factor

    tree = KDTree(xycells_aux)
    _, cell_number = tree.query(midpoints)

    return cell_number


def get_surface_mesh(ien):

    if ien.shape[1] == 3:   # Assuming tetra
        array = np.array([[0,1],[1,2],[2,0]])
        nelems = np.repeat(np.arange(ien.shape[0]),3)
        faces = np.vstack(ien[:,array])
        sort_faces = np.sort(faces,axis=1)

        f, i, c = np.unique(sort_faces, axis=0, return_counts=True, return_index=True)
        ind = i[np.where(c==1)[0]]
        bfaces = faces[ind]
        belem = nelems[ind]


    elif ien.shape[1] == 4:   # Assuming tetra
        array = np.array([[0,1,2],[1,2,3],[0,1,3],[2,0,3]])
        nelems = np.repeat(np.arange(ien.shape[0]),4)
        faces = np.vstack(ien[:,array])
        sort_faces = np.sort(faces,axis=1)

        f, i, c = np.unique(sort_faces, axis=0, return_counts=True, return_index=True)
        ind = i[np.where(c==1)[0]]
        bfaces = faces[ind]
        belem = nelems[ind]

    return belem, bfaces


def get_surface_normals(points, ien, vol_elems=None):
    points_elems = points[ien]
    if ien.shape[1] == 2:   # Lines
        v1 = points_elems[:,1] - points_elems[:,0]
        v2 = np.array([0,0,1])

        normal = np.cross(v1, v2, axisa=1)
        normal = normal/np.linalg.norm(normal,axis=1)[:,None]

    if ien.shape[1] == 3:

        v1 = points_elems[:,1] - points_elems[:,0]
        v2 = points_elems[:,2] - points_elems[:,0]

        normal = np.cross(v1, v2, axisa=1, axisb=1)
        normal = normal/np.linalg.norm(normal,axis=1)[:,None]

    if vol_elems is None:
        return normal

    elem_midpoint = np.mean(points[vol_elems], axis=1)
    face_midpoint = np.mean(points[ien], axis=1)

    vector = face_midpoint-elem_midpoint
    dot = np.sum(normal*vector, axis=1)
    normal[dot<0] *= -1

    return normal


def get_elem_neighbors(ien, cell_number):
    neigh_elems = np.zeros([len(ien), 3], dtype=int)
    for e in range(len(cell_number)):
        elem_nodes = ien[e]
        neigh = np.where(np.sum(np.isin(ien, elem_nodes), axis=1)==2)[0]
        neighbors = -np.ones(3, dtype=int)
        neighbors[0:len(neigh)] = neigh
        neigh_elems[e] = neighbors

    neigh_cells = np.zeros_like(neigh_elems) - 1
    neigh_cells[neigh_elems>-1] = cell_number[neigh_elems[neigh_elems>-1]]

    return neigh_elems, neigh_cells




def find_isolated_cells(cell_number, neigh_elems_og, neigh_cells):
    neigh_elems = np.copy(neigh_elems_og)

    nneighbors = 3-np.sum(neigh_cells==-1, axis=1)

    iso_cells = 10
    while iso_cells != 0:
        iso_cells = 0
        for e in range(len(cell_number)):
            if np.all(neigh_cells[e]==cell_number[e]): continue
            max_neighbors = nneighbors[e]
            non_cell_neigh = max_neighbors-np.sum(neigh_cells[e]==cell_number[e])

            if non_cell_neigh < 2: continue
            elems, counts = np.unique(neigh_cells[e], return_counts=True)
            cell_number[e] = elems[np.argmax(counts)]       # Assigning to cell with largest number of neighbors

            # update neigh_elems
            mask = neigh_elems[neigh_elems[e]]==e
            neigh_cells_elems = neigh_cells[neigh_elems[e]]
            neigh_cells_elems[mask] = cell_number[e]
            neigh_cells[neigh_elems[e]] = neigh_cells_elems

            # Only cells with no neighbors of their own cell pass
            if max_neighbors > non_cell_neigh: continue
            elems, counts = np.unique(neigh_cells[e], return_counts=True)
            cell_number[e] = elems[np.argmax(counts)]       # Assigning to cell with largest number of neighbors
            iso_cells += 1


    neigh_cells[neigh_elems>-1] = cell_number[neigh_elems[neigh_elems>-1]]

    return cell_number, neigh_cells


def make_discontinous_mesh(mesh, mask, map_new_nodes = None):
    xyz = mesh.points
    ien = mesh.cells[0].data

    cell_elems = np.where(mask)
    not_cell_elems = np.where(~mask)
    cell_nodes = np.unique(ien[cell_elems])
    not_cell_nodes = np.unique(ien[not_cell_elems])

    inter_nodes = np.intersect1d(cell_nodes, not_cell_nodes)
    inter_elems = np.where(np.any(np.isin(ien, inter_nodes), axis=1))[0]
    cell_elems = np.intersect1d(inter_elems, cell_elems)
    not_cell_elems = np.intersect1d(inter_elems, not_cell_elems)

    new_xyz = np.vstack([xyz, xyz[inter_nodes]])
    if map_new_nodes is None:
        map_new_nodes = np.arange(len(new_xyz))
    else:
        aux = np.arange(len(new_xyz))
        aux[:len(map_new_nodes)] = map_new_nodes
        map_new_nodes = aux
    map_new_nodes[inter_nodes] = np.arange(len(inter_nodes)) + len(xyz)
    map_aux = np.arange(len(new_xyz))
    map_aux[inter_nodes] = np.arange(len(inter_nodes)) + len(xyz)
    new_ien = np.copy(ien)
    new_ien[not_cell_elems] = map_aux[new_ien[not_cell_elems]]
    map_new_nodes[map_new_nodes[inter_nodes]] = inter_nodes

    new_mesh = io.Mesh(new_xyz, {'triangle': new_ien})

    return new_mesh, map_new_nodes


def generate_fully_disc_mesh(mesh, cell_number):
    map_new_nodes = None
    pressure_mesh = io.Mesh(mesh.points, mesh.cells)
    ncells = np.max(cell_number)
    for i in range(1, ncells):
        mask = cell_number == i
        pressure_mesh, map_new_nodes = make_discontinous_mesh(pressure_mesh, mask, map_new_nodes=map_new_nodes)

    pressure_mesh.cell_data['cell_number'] = [cell_number]

    # Need to generate a new bdata
    disc_belems, disc_blines = get_surface_mesh(pressure_mesh.cells[0].data)
    bxyz = pressure_mesh.points[disc_blines]
    b1nodes = np.where(np.sum(np.isclose(bxyz[:,:,0], 0), axis=1)==2)[0]
    b2nodes = np.where(np.sum(np.isclose(bxyz[:,:,0], np.max(bxyz[:,0])), axis=1)==2)[0]

    belems_disc = np.concatenate([b1nodes, b2nodes])
    arr = np.ones(len(belems_disc))
    arr[len(b1nodes):] = 2

    disc_bdata = np.vstack([disc_belems[belems_disc], disc_blines[belems_disc].T, arr]).T
    disc_bdata = disc_bdata.astype(int)

    # disc_mesh = io.Mesh(pressure_mesh.points, {'line': disc_blines})

    # disc_ien = pressure_mesh.cells[0].data
    # for i, e in enumerate(disc_bdata[:,0]):
    #     print(i, e, np.sum(np.isin(disc_ien[e], disc_bdata[i,1:-1])))

    return pressure_mesh, disc_bdata, map_new_nodes


def distance_point_hull(hull, p0):
    hull_lines = hull.points[hull.simplices]
    p1, p2 = np.swapaxes(hull_lines, 1, 0)

    vector = p0 - p1
    normal = p2 - p1
    normal = normal/np.linalg.norm(normal, axis=1)[:,None]
    aux = np.sum(vector*normal, axis=1)
    distance = np.linalg.norm(vector - aux[:,None]*normal, axis=1)

    return distance

def get_connected_boundary(points, max_angle):

    hull = ConvexHull(points)
    hull_normals = get_surface_normals(hull.points, hull.simplices)
    connected_node = np.zeros(len(points), dtype=bool)
    for i in range(len(points)):
        distance = distance_point_hull(hull, points[i])
        min_nodes = np.where(np.isclose(distance, np.min(distance)))[0]
        point_normal = hull_normals[min_nodes, 0:2]
        angle = np.arccos(np.abs(np.dot(point_normal, np.array([1,0]))))
        connected_node[i] = np.any(angle < max_angle)

    return connected_node


def find_connected_nodes(mesh, mask, connected_nodes=np.array([])):
    xyz = mesh.points
    ien = mesh.cells[0].data

    # Find nodes in angle
    cell_elems = np.where(mask)
    ien_cell = ien[cell_elems]
    tri_elem, line_ien = get_surface_mesh(ien_cell)
    line_nodes = np.unique(line_ien)
    points = xyz[line_nodes]
    connected_node = get_connected_boundary(points, np.pi/2*0.93)
    nodes_to_connect = line_nodes[connected_node]
    connected_nodes = np.append(connected_nodes, nodes_to_connect)

    return connected_nodes



def make_discontinous_continuous_mesh(mesh, mask, connected_nodes, map_new_nodes = None):
    xyz = mesh.points
    ien = mesh.cells[0].data

    # Find intersection nodes
    cell_elems = np.where(mask)
    not_cell_elems = np.where(~mask)
    cell_nodes = np.unique(ien[cell_elems])
    not_cell_nodes = np.unique(ien[not_cell_elems])
    inter_nodes = np.intersect1d(cell_nodes, not_cell_nodes)

    # Delete connected nodes from inter nodes
    inter_nodes = np.setdiff1d(inter_nodes, connected_nodes)

    # Find elements
    inter_elems = np.where(np.any(np.isin(ien, inter_nodes), axis=1))[0]
    cell_elems = np.intersect1d(inter_elems, cell_elems)
    not_cell_elems = np.intersect1d(inter_elems, not_cell_elems)

    new_xyz = np.vstack([xyz, xyz[inter_nodes]])
    if map_new_nodes is None:
        map_new_nodes = np.arange(len(new_xyz))
    else:
        aux = np.arange(len(new_xyz))
        aux[:len(map_new_nodes)] = map_new_nodes
        map_new_nodes = aux
    map_new_nodes[inter_nodes] = np.arange(len(inter_nodes)) + len(xyz)
    map_aux = np.arange(len(new_xyz))
    map_aux[inter_nodes] = np.arange(len(inter_nodes)) + len(xyz)
    new_ien = np.copy(ien)
    new_ien[not_cell_elems] = map_aux[new_ien[not_cell_elems]]
    map_new_nodes[map_new_nodes[inter_nodes]] = inter_nodes

    new_mesh = io.Mesh(new_xyz, {'triangle': new_ien})

    return new_mesh, map_new_nodes


def generate_connected_disc_mesh(mesh, cell_number, dsp_density=None):
    map_disp_new_nodes = None
    ncells = np.max(cell_number)

    connected_nodes = np.array([], dtype=int)
    disc_mesh = io.Mesh(mesh.points, mesh.cells)
    for i in range(1, ncells):
        mask = cell_number == i
        connected_nodes = find_connected_nodes(mesh, mask, connected_nodes)

    connected_nodes = np.unique(connected_nodes)
    if dsp_density is not None:
        dsp_connected = dsp_density[connected_nodes]
        connected = np.random.binomial(1, (dsp_connected+2)/3)
        connected_nodes = connected_nodes[connected==1]


    for i in range(1, ncells):
        mask = cell_number == i
        disc_mesh, map_new_nodes = make_discontinous_continuous_mesh(disc_mesh, mask, connected_nodes, map_new_nodes=map_disp_new_nodes)


    # TODO need to generate bdata
    return disc_mesh, connected_nodes, map_new_nodes


def prep_cheart():
    # Run cheart
    print('Running cheart')
    with open('prep.log', 'w') as ofile:
        p1 = Popen(['cheartsolver.out', 'prep.P', '--prep'],
                stdout=ofile, stderr=ofile)
        p2 = Popen(['cheartsolver.out', 'prep_cells.P', '--prep'],
                stdout=ofile, stderr=ofile)
    _ = [p.wait() for p in (p1, p2)]


def get_boundary_mesh(mesh, connected_nodes, disc_mesh):
    ien = mesh.cells[0].data
    _, boundary_faces = get_surface_mesh(ien)
    boundary_nodes = np.unique(boundary_faces)
    connected_nodes = np.setdiff1d(connected_nodes, boundary_nodes)

    _, cell_boundary = get_surface_mesh(disc_mesh.cells[0].data)
    cell_connected_boundary = np.sum(np.isin(cell_boundary, connected_nodes), axis=1)==2
    cell_connected_boundary = cell_connected_boundary.astype(int)

    cell_connected_nodes = np.zeros(len(disc_mesh.points), dtype=int)
    cell_connected_nodes[connected_nodes ] = 1
    cell_boundary_mesh = io.Mesh(disc_mesh.points, {'line': cell_boundary},
                                cell_data = {'connected': [cell_connected_boundary]},
                                point_data = {'connected': cell_connected_nodes})
    return cell_boundary_mesh