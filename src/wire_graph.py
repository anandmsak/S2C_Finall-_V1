import cv2
import numpy as np


def _to01(img):
    return (img > 0).astype(np.uint8)


# ---------------- NODE REGION LABELING ----------------

def label_node_regions_voronoi(nodes, shape, radius=10):
    h, w = shape[:2]
    node_lab = np.zeros((h, w), dtype=np.int32)

    if len(nodes) == 0:
        return node_lab, np.zeros((h, w), dtype=np.uint8)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    pts = np.array(nodes, dtype=np.float32)

    dist2 = (xx[None,:,:] - pts[:,0][:,None,None])**2 + (yy[None,:,:] - pts[:,1][:,None,None])**2
    nearest = np.argmin(dist2, axis=0)
    mind2 = np.min(dist2, axis=0)

    inside = mind2 <= radius*radius
    node_lab[inside] = nearest[inside] + 1
    node_mask = inside.astype(np.uint8)

    return node_lab, node_mask


# ---------------- SEGMENT ENDPOINTS ----------------

def segment_endpoints(seg):
    k = np.array([[1,1,1],[1,0,1],[1,1,1]], np.uint8)
    neigh = cv2.filter2D(seg, -1, k)
    ys, xs = np.where((seg==1) & (neigh==1))
    return list(zip(xs,ys))


def nearest_node(pt, nodes):
    if len(nodes)==0:
        return None, None
    x,y = pt
    pts = np.array(nodes, dtype=np.float32)
    d2 = (pts[:,0]-x)**2 + (pts[:,1]-y)**2
    i = np.argmin(d2)
    return int(i), float(d2[i])


# ---------------- BUILD GRAPH ----------------

def build_graph_from_skeleton(skel, endpoints, junctions,
                              node_radius=12,
                              min_seg_area=35,
                              endpoint_snap=22):

    sk = _to01(skel)
    nodes = list(endpoints) + list(junctions)

    node_lab, node_mask = label_node_regions_voronoi(nodes, sk.shape, node_radius)

    sk_wo_nodes = sk.copy()
    sk_wo_nodes[node_mask==1] = 0

    num, labels = cv2.connectedComponents(sk_wo_nodes, 8)

    edges = []
    snap_r2 = endpoint_snap * endpoint_snap

    for seg_id in range(1, num):
        ys, xs = np.where(labels == seg_id)
        if len(xs) < min_seg_area:
            continue

        seg = np.zeros_like(sk)
        seg[ys, xs] = 1

        ends = segment_endpoints(seg)
        if len(ends) == 0:
            continue

        candidates = []
        for p in ends:
            ni, d2 = nearest_node(p, nodes)
            if ni is not None and d2 <= snap_r2:
                candidates.append((d2, ni))

        if len(candidates) < 2:
            continue

        candidates.sort()
        chosen = []
        for _, ni in candidates:
            if ni not in chosen:
                chosen.append(ni)
            if len(chosen) == 2:
                break

        if len(chosen)==2 and chosen[0]!=chosen[1]:
            u,v = sorted(chosen)
            edges.append((u,v,list(zip(xs,ys))))

    # remove duplicates
    uniq={}
    for u,v,pix in edges:
        key=(u,v)
        if key not in uniq or len(pix)>len(uniq[key]):
            uniq[key]=pix

    edges_final=[(u,v,uniq[(u,v)]) for (u,v) in uniq]

    return nodes, edges_final, (sk_wo_nodes*255).astype(np.uint8)


# ---------------- DEBUG DRAW ----------------

def draw_graph_debug(skel, nodes, edges):
    vis = cv2.cvtColor((_to01(skel)*255), cv2.COLOR_GRAY2BGR)

    for u,v,pix in edges:
        for x,y in pix:
            vis[y,x]=(0,255,0)

    for i,(x,y) in enumerate(nodes):
        cv2.circle(vis,(int(x),int(y)),5,(0,0,255),-1)
        cv2.putText(vis,str(i),(int(x)+6,int(y)-6),
                    cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1)

    return vis


# ---------------- NODE MERGE ----------------

def merge_close_nodes(nodes, merge_radius=40):
    pts=np.array(nodes,np.float32)
    n=len(pts)
    if n==0: return [],[]

    parent=list(range(n))
    def find(a):
        while parent[a]!=a:
            parent[a]=parent[parent[a]]
            a=parent[a]
        return a
    def union(a,b):
        pa,pb=find(a),find(b)
        if pa!=pb: parent[pb]=pa

    r2=merge_radius*merge_radius
    for i in range(n):
        for j in range(i+1,n):
            if np.sum((pts[i]-pts[j])**2)<=r2:
                union(i,j)

    clusters={}
    for i in range(n):
        clusters.setdefault(find(i),[]).append(i)

    merged=[]
    mapping=[0]*n
    for nid,idxs in enumerate(clusters.values()):
        cx=np.mean(pts[idxs,0])
        cy=np.mean(pts[idxs,1])
        merged.append((cx,cy))
        for k in idxs:
            mapping[k]=nid

    return merged,mapping


def remap_edges(edges,mapping):
    uniq={}
    for u,v,pix in edges:
        u2,v2=mapping[u],mapping[v]
        if u2==v2: continue
        if u2>v2: u2,v2=v2,u2
        key=(u2,v2)
        if key not in uniq or len(pix)>len(uniq[key]):
            uniq[key]=pix
    return [(u,v,uniq[(u,v)]) for (u,v) in uniq]


# ---------------- DEGREE-2 SIMPLIFY ----------------

def compress_degree2(nodes, edges):
    n=len(nodes)
    adj={i:set() for i in range(n)}
    pixmap={}
    for u,v,p in edges:
        adj[u].add(v); adj[v].add(u)
        pixmap[(min(u,v),max(u,v))]=p

    removed=set()
    changed=True
    while changed:
        changed=False
        for k in range(n):
            if k in removed: continue
            if len(adj[k])==2:
                a,b=list(adj[k])
                key1=(min(a,k),max(a,k))
                key2=(min(b,k),max(b,k))
                p1=pixmap.pop(key1,[])
                p2=pixmap.pop(key2,[])
                new=(min(a,b),max(a,b))
                pixmap[new]=p1+p2
                adj[a].remove(k); adj[b].remove(k)
                adj[a].add(b); adj[b].add(a)
                removed.add(k)
                changed=True
                break

    keep=[i for i in range(n) if i not in removed]
    remap={old:i for i,old in enumerate(keep)}

    new_nodes=[nodes[i] for i in keep]
    new_edges=[(remap[u],remap[v],pixmap[(u,v)]) for (u,v) in pixmap if u in remap and v in remap]

    return new_nodes,new_edges
