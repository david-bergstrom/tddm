import numpy as np

def crop_signal_with_rotation(float[:, :] signal, float[:] region, int min_len):
    cdef int i
    
    cdef float origin_lat = region[0]
    cdef float origin_lon = region[1]
    cdef float rotation = region[2]
    cdef float height = region[3]
    cdef float width = region[4]
    
    cdef float lat
    cdef float lon

    rotation = rotation * np.pi / 180

    cdef float cosrotation = np.cos(rotation)
    cdef float sinrotation = np.sin(rotation)
    cdef float cosnrotation = np.cos(-rotation)
    cdef float sinnrotation = np.sin(-rotation)

    rotated_signal = np.empty_like(signal)
    # rotated_signal[:, 0] = np.cos(rotation) * signal[:, 0] + np.sin(-rotation) * signal[:, 1]
    # rotated_signal[:, 1] = np.sin(rotation) * signal[:, 0] + np.cos(-rotation) * signal[:, 1]

    parts = []
    last_outside_region = -1
    
    for i in range(len(signal)):
        lat = signal[i, 0]
        lon = signal[i, 1]

        lat -= origin_lat
        lon -= origin_lon

        lat /= height
        lon /= width

        rotated_signal[i, 0] = cosnrotation * lat + sinnrotation * lon
        rotated_signal[i, 1] = sinrotation * lat + cosnrotation * lon
        lat = rotated_signal[i, 0]
        lon = rotated_signal[i, 1]

        if lat < 0 or lat > 1 \
            or lon < 0 or lon > 1:
            
            part = rotated_signal[last_outside_region+1:i]

            if len(part) >= min_len:
                parts.append(part)
            
            last_outside_region = i

    
    part = rotated_signal[last_outside_region+1:i+1]

    if len(part) >= min_len:
        parts.append(part)
    
    last_outside_region = i
                    
    return parts