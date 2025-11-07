# Franka Collision Analyzer Issue - Analysis and Solutions

## Problem Summary

The collision analyzer fails for the Franka robot because `self.local_pts` is empty when trying to concatenate, indicating that no collision meshes are found under the robot's body links.

## Detailed Structural Differences Between UR5e and Franka USDs

### UR5e USD Structure (Working)

**File:** `ur5e_robotiq_readable.usda`

**Structure for each body link (e.g., `wrist_3_link`):**
```usda
def Xform "wrist_3_link" (
    apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
)
{
    # Physics properties...
    
    def Xform "collisions" (
        instanceable = true
        add references = </Flattened_Prototype_12>  # Internal reference within same file
    )
    {
        # This Xform contains collision meshes directly accessible
    }
}
```

**Key Characteristics:**
1. **Internal References**: Uses `add references = </Flattened_Prototype_12>` pointing to a prim defined earlier in the same USD file
2. **Direct Access**: The `Flattened_Prototype_12` prim contains collision meshes (Mesh prims with `PhysicsCollisionAPI`) as direct children
3. **File Structure**: Everything is in one USD file - no external file dependencies
4. **Traversal**: `get_all_matching_child_prims` can find colliders because they're in the same stage hierarchy

### Franka USD Structure (Failing)

**Main File:** `franka_instanceable_readable.usda`  
**Collision File:** `franka_collisions.usd` (separate file)

**Structure for each body link (e.g., `panda_link7`):**
```usda
def Xform "panda_link7" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
)
{
    # Physics properties...
    
    def "collisions" (
        instanceable = true
        references = @./franka_collisions.usd@</panda_link7_collisions>  # External file reference
    )
    {
        # Empty - content is in referenced file
    }
}
```

**Referenced File (`franka_collisions.usd`):**
```usda
over Xform "panda_link7_collisions"
{
    def Mesh "collisions" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
    )
    {
        # Collision mesh data (vertices, faces, etc.)
    }
}
```

**Key Characteristics:**
1. **External File References**: Uses `references = @./franka_collisions.usd@</panda_link7_collisions>` pointing to a separate USD file
2. **Separated Assets**: Collision meshes are in a different file (`franka_collisions.usd`) from the main robot file
3. **File Structure**: Multi-file architecture with references between files
4. **Traversal Issue**: `get_all_matching_child_prims` may not traverse through external file references correctly

### Why This Causes the Problem

The `RigidObjectHasher` class uses `get_all_matching_child_prims` to find colliders:

```python
coll_prims = prim_utils.get_all_matching_child_prims(
    prim_paths[i],
    predicate=lambda p: ... and p.HasAPI(UsdPhysics.CollisionAPI),
)
```

**Current Behavior:**
- For UR5e: Finds colliders because they're in the same stage hierarchy (internal references are resolved)
- For Franka: Returns empty list because `get_all_matching_child_prims` may not traverse through external file references

**USD Reference Resolution:**
- USD references ARE automatically loaded when the stage is opened
- However, `GetChildren()` or `GetFilteredChildren(Usd.TraverseInstanceProxies())` may not traverse through references
- The referenced content exists in the stage, but the traversal method may skip it

## Solution Approaches

### Approach 1: Modify Collision Analyzer to Handle References

**What to do:**
Modify `RigidObjectHasher` to use a traversal method that follows USD references.

**Implementation:**

1. **Option 1A: Use `UsdPrim::GetAllChildren()` with reference traversal**
   - Modify `get_all_matching_child_prims` to accept a parameter for traversing references
   - Use `Usd.Prim.GetAllChildren()` or a custom traversal that follows references
   - Check if references are already resolved and traverse them

2. **Option 1B: Manually resolve references before traversal**
   - Before calling `get_all_matching_child_prims`, check if body prim has a `collisions` child with a reference
   - If so, resolve the reference and search under the referenced prim path
   - Example code:
   ```python
   # In collision_analyzer.py or rigid_object_hasher.py
   body_prim = stage.GetPrimAtPath(body_prim_path)
   collisions_prim = body_prim.GetChild("collisions")
   if collisions_prim and collisions_prim.HasAuthoredReferences():
       # Get the referenced prim path
       refs = collisions_prim.GetReferences()
       # Traverse the referenced content
       coll_prims = get_all_matching_child_prims(
           str(collisions_prim.GetPath()),  # Search under collisions
           predicate=lambda p: p.HasAPI(UsdPhysics.CollisionAPI),
       )
   ```

3. **Option 1C: Use `UsdPrim::GetFilteredChildren()` with custom filter**
   - Create a custom filter that includes references
   - Use `Usd.Prim.GetFilteredChildren()` with a filter that traverses references

**Pros:**
- Fixes the root cause
- Works for any USD structure (internal or external references)
- No changes needed to USD files

**Cons:**
- Requires understanding USD reference traversal
- May need to test edge cases
- More complex implementation

### Approach 2: Ensure References Are Loaded/Flattened

**What to do:**
Ensure that when the Franka USD is loaded, all references are expanded/flattened so colliders are directly accessible.

**Implementation:**

1. **Option 2A: Flatten USD at asset spawn time**
   - Modify the asset spawner to flatten references when loading the Franka USD
   - Use `UsdStage.Flatten()` or similar to inline all references
   - This would create a single USD file with all content inlined

2. **Option 2B: Use `UsdStage.Flatten()` in collision analyzer**
   - Before searching for colliders, temporarily flatten the stage
   - Search for colliders in the flattened stage
   - This is a workaround, not recommended for production

3. **Option 2C: Modify USD files to inline references**
   - Use USD tools to flatten `franka_instanceable.usd` and `franka_collisions.usd` into a single file
   - Update the config to point to the flattened file
   - This matches UR5e's structure

**Pros:**
- Simple - just change USD file structure
- No code changes needed
- Matches UR5e's working structure

**Cons:**
- Requires USD file manipulation
- May lose benefits of separate files (modularity, file size)
- Need to regenerate USD files if originals change

## Recommended Solution: Approach 2C (Flatten USD Files)

**Why this is easiest:**
1. **No code changes needed** - The collision analyzer already works for UR5e's structure
2. **Quick to implement** - Just flatten the USD files using USD tools
3. **Matches existing working pattern** - UR5e uses this structure successfully
4. **Low risk** - No modifications to complex traversal logic

**How to implement:**

1. **Flatten the Franka USD files:**
   ```python
   # Using USD Python API
   from pxr import Usd, UsdStage
   
   # Open the main file
   main_stage = UsdStage.Open("/path/to/franka_instanceable.usd")
   
   # Flatten to remove references (inlines everything)
   flattened_stage = main_stage.Flatten()
   
   # Save the flattened version
   flattened_stage.Export("/path/to/franka_instanceable_flattened.usd")
   ```

2. **Update the config:**
   ```python
   # In franka.py
   FRANKA_ROBOTIQ_GRIPPER_CUSTOM_OMNI_PAT_CFG.spawn.usd_path = "/path/to/franka_instanceable_flattened.usd"
   ```

3. **Verify structure:**
   - After flattening, check that `panda_link7/collisions` contains the Mesh directly
   - The structure should match UR5e's pattern

**Alternative Quick Fix (Approach 1B):**

If flattening isn't preferred, a minimal code change in `rigid_object_hasher.py`:

```python
# In rigid_object_hasher.py, around line 48
for i in range(num_roots):
    # Try to find colliders under the body prim
    body_prim_path = prim_paths[i]
    body_prim = prim_utils.get_prim_at_path(body_prim_path)
    
    # Check if there's a "collisions" child with a reference
    collisions_prim = body_prim.GetChild("collisions")
    if collisions_prim and collisions_prim.HasAuthoredReferences():
        # Search under the collisions prim (references are resolved)
        search_path = str(collisions_prim.GetPath())
    else:
        # Search under the body prim directly
        search_path = body_prim_path
    
    coll_prims = prim_utils.get_all_matching_child_prims(
        search_path,  # Use collisions path if it exists
        predicate=lambda p: prim_utils.get_prim_at_path(p).GetTypeName()
        in ("Mesh", "Cube", "Sphere", "Cylinder", "Capsule", "Cone")
        and prim_utils.get_prim_at_path(p).HasAPI(UsdPhysics.CollisionAPI),
    )
```

This checks for a `collisions` child and searches under it, which should find the referenced mesh.

## Recommendation

**Easiest path forward: Flatten the Franka USD files** (Approach 2C)

This requires:
1. Running a USD flattening script once
2. Updating the config path
3. No code changes
4. Matches the working UR5e structure

If you want to keep the modular USD structure, use **Approach 1B** with the minimal code change above.

