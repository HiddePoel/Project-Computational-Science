import numpy as np
import init
import verlet
import satellites
from find_paths import iterate_permutations
# from numba import njit, jit  (TODO)


SNAPSHOTS_DIR = "../snapshots/"


if __name__ == "__main__":
    planets_pos, planets_vel, planets_mass = init.planets()
    sats_pos, sats_vel, goes_idx = init.satellites(noDownload=True)
    n_sats = len(sats_pos)

    # Need to set this to whatever our start time is when initialising
    t0 = 0

    # (seconds)
    dt = 100

    # (km)
    sat_opening_thresh = 150

    # INIT VISUALISER HERE (TODO)
    ...
    # VISUALISE INITIAL POSITIONS HERE
    ...

    print("-------------------------------------------------------------------")
    print("Start looking for candidate launch times.")
    for step in range(50):
        planets_pos, planets_vel = verlet.update(planets_pos, planets_vel,
                                                 planets_mass, dt, G=6.674e-11)

        # TODO optimisation
        sats_pos, sats_vel = satellites.update(sats_pos, sats_vel, planets_mass, dt)

        # Test Multithreading
        # with ThreadPoolExecutor() as executor:
        #     # Launch multithreaded computation for all satellites
        #     futures = [
        #         executor.submit(update_satellite, sat, sats_pos, sats_vel, planets_mass, dt)
        #         for sat in range(n_sats)
        #     ]

        #     # Collect the results
        #     for future in futures:
        #         sat, new_pos, new_vel = future.result()
        #         sats_pos[sat] = new_pos
        #         sats_vel[sat] = new_vel

        # VISUALISE UPDATES POS' HERE (TODO)
        ...

        pos_launch, point_above = satellites.launch_site(planets_pos, sats_pos, goes_idx)
        launch_normal = point_above - pos_launch

        # Checks for a candidate launch time.
        opening = satellites.opening(sats_pos, pos_launch, launch_normal, goes_idx)
        # print(pos_launch, point_above, opening) #(DEBUG)
        if sat_opening_thresh < opening:
            print("-------------------------------------------------------------------")
            print("Candidate launch time found at init time + ", step * dt, " seconds.")
            print("Opening: ", opening, " km")

            # Save current permutation to a file
            path = SNAPSHOTS_DIR + "t" + str(step)
            np.savez(path, planets_pos=planets_pos,
                     planets_vel=planets_vel,
                     launch_normal=launch_normal + planets_pos[2])

            print("Snapshot saved at ", path)

    iterate_permutations()
