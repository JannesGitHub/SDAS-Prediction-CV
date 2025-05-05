import os
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from glob import glob
from cellpose import core, utils, io, models, metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.spatial import cKDTree

def draw_contours_with_alpha(img, masks, alpha=0.1, color=(255, 0, 0)):
    """
    Draw contours on the image with alpha transparency and display it using matplotlib.

    Parameters:
    - img: The image to draw contours on.
    - masks: The masks containing the segmented regions to be outlined.
    - alpha: Transparency level for the contours (0: fully transparent, 1: fully opaque).
    - color: Color of the contours in RGB format (default is red).
    """
    img_contours = img.copy() #img_contours soll auch diese Linien die hier gezeichnet werden erhalten

    # Konturen zeichnen
    outlines = utils.outlines_list(masks)
    for o in outlines:
        # Koordinaten in Integer umwandeln (OpenCV erwartet Integer)
        points = o.astype(np.int32)
        
        # Linien mit OpenCV zeichnen (Farbe: Rot [RGB], Alpha für Transparenz)
        for i in range(len(points) - 1):
            cv2.line(img_contours, tuple(points[i]), tuple(points[i+1]), color, 1) #wie kann ich hier alpha einstellen also die Deckkraft der Linie
        # Transparenz simulieren (Linien mit Alpha-Überblendung)
        img_contours = cv2.addWeighted(img_contours, 1, img_contours, 0, 0)
        
     # Simulate transparency with alpha blending
    img_contours = cv2.addWeighted(img_contours, 1 - alpha, img_contours, alpha, 0)

    # Bild mit Konturen anzeigen
    plt.figure(figsize=(8,12))
    plt.imshow(img_contours)
    plt.axis("off")
    plt.title("Segmentierung")

    plt.tight_layout()
    plt.show()
    
    return img_contours

def extract_line_segments(masks):
    """
    Extracts line segments and their corresponding contours from segmented objects in the mask.

    Parameters:
    - masks: A 2D array where each unique non-zero value corresponds to a segmented object.

    Returns:
    - line_segments: A list of tuples representing line segments.
      Each tuple contains:
      (start_x, start_y, end_x, end_y, angle, mid_x, mid_y, contour)
    """
    line_segments = []
    
    for obj_id in np.unique(masks):
        if obj_id == 0:
            continue

        # Create binary mask for this object
        mask = (masks == obj_id).astype(np.uint8)

        # Find contours using OpenCV
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue

        # Assume largest contour corresponds to the object
        contour = max(contours, key=cv2.contourArea).squeeze()
        if contour.ndim != 2 or contour.shape[0] < 2:
            continue

        # Extract pixel coordinates from contour
        x_coords, y_coords = contour[:, 0], contour[:, 1]
        data = np.column_stack((x_coords, y_coords))
        mean = data.mean(axis=0)
        data_centered = data - mean
        cov = np.cov(data_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        primary_vec = eigenvectors[:, np.argmax(eigenvalues)]

        # Find extreme points along the principal axis
        projections = np.dot(data_centered, primary_vec)
        pt_min = mean + projections.min() * primary_vec
        pt_max = mean + projections.max() * primary_vec

        raw_angle = np.degrees(np.arctan2(primary_vec[1], primary_vec[0]))
        angle = ((raw_angle + 90) % 180) - 90

        mid_x = (pt_min[0] + pt_max[0]) / 2
        mid_y = (pt_min[1] + pt_max[1]) / 2

        # Save everything including the contour
        line_segments.append((
            int(pt_min[0]), int(pt_min[1]),
            int(pt_max[0]), int(pt_max[1]),
            angle, mid_x, mid_y,
            contour  # <-- Die Kontur wird hier mitgegeben
        ))

    return line_segments

def plot_line_segments_on_image(img, line_segments):
    """
    Plots the extracted line segments on the image using matplotlib.
    
    Parameters:
    - img: The image on which to plot the line segments.
    - line_segments: A list of tuples representing line segments.
      Each tuple contains (start_x, start_y, end_x, end_y, angle, mid_x, mid_y).
    """
    
    img_lines = img.copy()
    
    plt.figure(figsize=(8, 12))
    plt.imshow(img)
    plt.axis("off")
    
    # Plot each line segment
    for line in line_segments:
        start_x, start_y, end_x, end_y, angle, mid_x, mid_y, contours = line
        
        # draw that line on img_lines
        pt1 = (int(round(start_x)), int(round(start_y)))
        pt2 = (int(round(end_x)), int(round(end_y)))
        cv2.line(img_lines, pt1, pt2, (0,0,255), 3)
        
        plt.plot([start_x, end_x], [start_y, end_y], color='red', linewidth=2)  # Plot lines in red

    plt.title("Line Segments Overlay")
    plt.tight_layout()
    plt.show()
    
    return img_lines

def plot_line_midpoints_with_angles(line_segments, img_lines):
    """
    Plots a scatter plot of the midpoints of line segments with color representing the angle.
    
    Parameters:
    - line_segments: A list of tuples representing line segments.
      Each tuple contains (start_x, start_y, end_x, end_y, angle, mid_x, mid_y).
    - save_dir: Directory where the plot will be saved.
    """
    filtered_segments = [seg for seg in line_segments if -90 <= seg[4] <= 90] # Hier kann man Winkelbereich einstellen für Filterung

    # Extrahiere Winkel und Mittelpunkte für das Diagramm
    angles = [seg[4] for seg in filtered_segments]
    mid_xs = [seg[5] for seg in filtered_segments]
    mid_ys = [seg[6] for seg in filtered_segments]

    # Scatterplot: Mittelpunkte (x, y) mit Farbe entsprechend des Winkels
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(img_lines, cv2.COLOR_BGR2RGB))
    sc = plt.scatter(mid_xs, mid_ys, c=angles, cmap='twilight', alpha=0.7, edgecolors='b')
    plt.xlabel("X-Koordinate")
    plt.ylabel("Y-Koordinate")
    plt.title("Overlapped")
    plt.colorbar(sc, label="Winkel (Grad)")

    plt.grid(True)
    plt.show()

def sample_contour(contour, n=50):
    idx = np.linspace(0, len(contour) - 1, n).astype(int)
    return contour[idx]

def distanceForLines(lineA, lineB):
    """
    Effiziente Berechnung:
    - Minimaler euklidischer Abstand zwischen den Konturen (per KDTree)
    - Winkelunterschied
    """
    angle_A = lineA[4]
    contour_A = lineA[7]

    angle_B = lineB[4]
    contour_B = lineB[7]

    # Nimm jeweils nur N Punkte aus den Konturen zur Berechnung der Distanz der Konturen um die Laufzeit zu minimieren (gleichverteilte Abstände)
    contour_A = sample_contour(lineA[7], 25) 
    contour_B = sample_contour(lineB[7], 25)

    angle_diff = min(abs(angle_A - angle_B), 180 - abs(angle_A - angle_B))

    # Erstelle KDTree für eine der Konturen (z. B. B)
    tree = cKDTree(contour_B)

    # Für jeden Punkt in A, finde den nächsten Punkt in B
    dists, _ = tree.query(contour_A, k=1)

    # Nimm den kleinsten dieser Abstände
    diff = np.min(dists)

    return diff, angle_diff

def fit_regression_line(points):
    """
    Schätzt eine lineare Regression basierend auf den Mittelpunkten (mid_x, mid_y) der Punkte.
    Ignoriert dabei die Kontur-Information (letztes Element in jedem Tupel).
    
    Returns:
    - reg: Lineares Regressionsmodell
    - variance: Mittlere quadratische Abweichung (Varianz der Residuen)
    """
    # Entferne das letzte Element (Kontur) aus jedem Tupel
    core_points = [p[:7] for p in points]  # nur die ersten 7 Elemente behalten (x1, y1, x2, y2, angle, mid_x, mid_y)
    points_arr = np.array(core_points)

    # Regressionsdaten: mid_x (Spalte 5) und mid_y (Spalte 6)
    X = points_arr[:, 5].reshape(-1, 1)
    y = points_arr[:, 6]

    # Lineare Regression durchführen
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)

    # Mittlere quadratische Abweichung (MSE)
    variance = np.mean((y - y_pred)**2)
    return reg, variance

def point_line_distance(point, reg):
    """
    Berechnet den senkrechten Abstand eines Punktes zur Regressionsgeraden.
    Hier werden die Mittelpunkte (mid_x, mid_y) verwendet:
      - point[5] entspricht mid_x
      - point[6] entspricht mid_y
    """
    m = reg.coef_[0]
    b = reg.intercept_
    x0 = point[5]
    y0 = point[6]
    return abs(m * x0 - y0 + b) / math.sqrt(m**2 + 1)

def calculate_sdas(line_data, mikrometer_per_pixel): #mit Kallibrierungsfaktor
    """
    Berechnet den SDAS-Wert für eine gegebene Linie.
    Dabei wird zunächst die maximale euklidische Distanz zwischen allen Paaren
    der Mittelpunkte (Spalten 5 und 6) berechnet und durch (#datenpunkte - 1) geteilt.
    """
    midpoints = np.array([(seg[5], seg[6]) for seg in line_data])
    max_dist = 0
    n = len(midpoints)
    for j in range(n):
        for k in range(j + 1, n):
            dist = np.linalg.norm(midpoints[j] - midpoints[k])
            if dist > max_dist:
                max_dist = dist
    # Vermeide Division durch 0, falls n == 1 (sollte aber nicht auftreten, da MIN_POINTS_PER_LINE >= 4)
    return (max_dist / (n - 1))*mikrometer_per_pixel if n > 1 else 0

def group_line_segments(line_segments, MAX_POINTS_PER_LINE, MAX_ANGLE_DIFF_REG_P_THRESHOLD,
                         REGRESSION_DISTANCE_THRESHOLD, ANGLE_DIFF_THRESHOLD, DISTANCE_THRESHOLD,
                         MIN_POINTS_PER_LINE, MICROMETER_PER_PIXEL):
    """
    Groups line segments based on proximity and angle to form continuous lines.

    Parameters:
    - ungrouped_points: List of points to be grouped into lines.
    - MAX_POINTS_PER_LINE: Maximum number of points allowed in a line.
    - MAX_ANGLE_DIFF_REG_P_THRESHOLD: Maximum allowed angle difference for regression-based filtering.
    - REGRESSION_DISTANCE_THRESHOLD: Maximum allowed distance for points to be added to the same line.
    - ANGLE_DIFF_THRESHOLD: Maximum allowed angle difference to consider a point for the line.
    - DISTANCE_THRESHOLD: Maximum allowed distance between points to add to the line.
    - MIN_POINTS_PER_LINE: Minimum number of points required to form a line.
    - MICROMETER_PER_PIXEL: Conversion factor for calculating SDAS.

    Returns:
    - lines: List of grouped lines, where each line contains a list of points and associated line parameters.
    """

    line_segments = line_segments.copy() # verhindert, dass die line_segments geändert werden.

    lines = []  # Final list of grouped lines
    i = 0       # Index for iteration

    while len(line_segments) > 0: #iteriert durch jede Kontur
        if i >= len(line_segments):
            break

        p_start = line_segments[i] #Nimm die nächste Kontur 
        current_line = [p_start] #Erstelle eine Gruppe von Konturen mit der Startkontur drin

        while len(current_line) < MAX_POINTS_PER_LINE:
            candidates = []
            last_point = current_line[-1]
            
            for p in line_segments:
                if p in current_line:
                    continue

                diff, angle_diff = distanceForLines(last_point, p)

                if angle_diff < ANGLE_DIFF_THRESHOLD and diff < DISTANCE_THRESHOLD:
                    candidates.append((p, diff, angle_diff))

            if not candidates:
                break
            
            # Jetzt Regression berechnen, da wir potenzielle Kandidaten haben
            if len(current_line) >= 2:
                reg, variance = fit_regression_line(current_line)
                angle_reg = math.degrees(math.atan(reg.coef_[0]))
            else:
                reg = None

            best_candidate = None
            best_metric = float('inf')
            
            for p, diff, angle_diff in candidates:
                if reg:
                    reg_distance = point_line_distance(p, reg)
                    angle_diff_to_reg = min(abs(p[4] - angle_reg), 180 - abs(p[4] - angle_reg))

                    if angle_diff_to_reg < MAX_ANGLE_DIFF_REG_P_THRESHOLD or reg_distance > REGRESSION_DISTANCE_THRESHOLD:
                        continue

                # Greedy-Kriterium: minimaler Abstand
                metric = diff + angle_diff  # du kannst das gerne anpassen oder gewichten
                if metric < best_metric:
                    best_candidate = p
                    best_metric = metric

            if best_candidate is None:
                break

            current_line.append(best_candidate)
            line_segments.remove(best_candidate)

        # If the line has enough points, calculate SDAS and store the line
        if len(current_line) >= MIN_POINTS_PER_LINE:
            sdas = calculate_sdas(current_line, MICROMETER_PER_PIXEL)
            # Store the line with the regression model, variance, and SDAS
            lines.append((current_line, reg if len(current_line) >= 2 else None,
                          variance if len(current_line) >= 2 else None, sdas))

        i += 1

    return lines

from scipy.spatial import cKDTree
import math
import numpy as np

def group_line_segments_with_KDTree(line_segments, MAX_POINTS_PER_LINE, MAX_ANGLE_DIFF_REG_P_THRESHOLD,
                         REGRESSION_DISTANCE_THRESHOLD, ANGLE_DIFF_THRESHOLD, DISTANCE_THRESHOLD,
                         MIN_POINTS_PER_LINE, MICROMETER_PER_PIXEL, KDTREEDISTANCE=500):
    """
    Groups line segments based on proximity and angle to form continuous lines.
    """

    line_segments = line_segments.copy()
    lines = []

    # Precompute centers for all contours
    centers = [np.mean(p[7], axis=0) for p in line_segments]
    tree = cKDTree(centers)

    used_indices = set()

    for i in range(len(line_segments)):
        if i in used_indices:
            continue

        p_start = line_segments[i]
        current_line = [p_start]
        used_indices.add(i)

        while len(current_line) < MAX_POINTS_PER_LINE:
            last_point = current_line[-1]
            last_center = np.mean(last_point[7], axis=0)

            # Find neighbors in distance
            idxs = tree.query_ball_point(last_center, KDTREEDISTANCE)

            candidates = []
            for idx in idxs:
                if idx in used_indices:
                    continue

                p = line_segments[idx]
                diff, angle_diff = distanceForLines(last_point, p)

                if angle_diff < ANGLE_DIFF_THRESHOLD and diff < DISTANCE_THRESHOLD:
                    candidates.append((idx, p, diff, angle_diff))

            if not candidates:
                break

            # Optional regression
            if len(current_line) >= 2:
                reg, variance = fit_regression_line(current_line)
                angle_reg = math.degrees(math.atan(reg.coef_[0]))
            else:
                reg = None

            best_idx = None
            best_metric = float('inf')

            for idx, p, diff, angle_diff in candidates:
                if reg:
                    reg_distance = point_line_distance(p, reg)
                    angle_diff_to_reg = min(abs(p[4] - angle_reg), 180 - abs(p[4] - angle_reg))
                    if angle_diff_to_reg < MAX_ANGLE_DIFF_REG_P_THRESHOLD or reg_distance > REGRESSION_DISTANCE_THRESHOLD:
                        continue

                metric = diff + angle_diff
                if metric < best_metric:
                    best_idx = idx
                    best_metric = metric

            if best_idx is None:
                break

            current_line.append(line_segments[best_idx])
            used_indices.add(best_idx)

        if len(current_line) >= MIN_POINTS_PER_LINE:
            sdas = calculate_sdas(current_line, MICROMETER_PER_PIXEL)
            lines.append((
                current_line,
                reg if len(current_line) >= 2 else None,
                variance if len(current_line) >= 2 else None,
                sdas
            ))

    return lines

def print_result_lines_over_img(lines, img):
    plt.figure(figsize=(8, 6)) #Hier ist der Fehler

    # Hintergrundbild anzeigen
    plt.imshow(img)

    # Definiere eine Farbpalette mit so vielen Farben wie Linien vorhanden sind
    colors = plt.cm.get_cmap("tab10", len(lines))

    # Iteriere über jede Linie (bestehend aus (line_data, reg, var, sdas))
    for i, (line_data, reg, var, sdas) in enumerate(lines):
        # Extrahiere die Mittelpunkte (mid_x, mid_y) aus jedem Segment der Linie
        midpoints = np.array([(seg[5], seg[6]) for seg in line_data])

        # Initialisiere Variablen für die maximale Distanz und die Endpunkte
        max_dist = 0
        pt1, pt2 = None, None

        # Berechne die euklidische Distanz für alle Punktpaare und finde das maximale Paar
        for j in range(len(midpoints)):
            for k in range(j + 1, len(midpoints)):
                dist = np.linalg.norm(midpoints[j] - midpoints[k])
                if dist > max_dist:
                    max_dist = dist
                    pt1, pt2 = midpoints[j], midpoints[k]

        # Zeichne die Hauptlinie zwischen den beiden entferntesten Punkten
        if pt1 is not None and pt2 is not None:
            plt.plot(
                [pt1[0], pt2[0]], [pt1[1], pt2[1]],
                color=colors(i), linewidth=2,
                label=f'Hauptlinie {i + 1} (SDAS={sdas:.3f})'
            )

    # Achseneinstellungen
    plt.xlabel("X-Koordinate")
    plt.ylabel("Y-Koordinate")
    plt.title("Hauptlinien: Verbindung der zwei entferntesten Punkte pro Linie")

    # Korrektes Seitenverhältnis einstellen
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')  # Gleichmäßige Skalierung

    # Legende positionieren
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)

    plt.show()

def calculate_avg_sdas(lines):
    """
    Calculate the average SDAS from the third index of each line tuple in the lines list.

    Parameters:
    - lines: A list of tuples where the third element (index 3) is the SDAS value.

    Returns:
    - avg_sdas: The average SDAS value.
    """
    avg_sdas = 0

    # Sum the SDAS values from index 3 of each line tuple
    for line in lines:
        avg_sdas += line[3]

    # Calculate the average SDAS
    avg_sdas /= len(lines) if len(lines) > 0 else 1  # Prevent division by zero

    return avg_sdas

def getResults(test_dir, model, diameter, MAX_POINTS_PER_LINE, MAX_ANGLE_DIFF_REG_P_THRESHOLD,
               REGRESSION_DISTANCE_THRESHOLD, ANGLE_DIFF_THRESHOLD, DISTANCE_THRESHOLD,
               MIN_POINTS_PER_LINE, MICROMETER_PER_PIXEL):
    
    results = []
    img_files = [f for f in os.listdir(test_dir) if f.endswith('.jpg') or f.endswith('.png')]
    total = len(img_files)

    for i, img_name in enumerate(img_files):
        # Gruppiertes Print alle 5 Bilder
        if i % 5 == 0:
            batch_files = img_files[i:i+5]
            print(f"Verarbeite Bilder {min(i+5, total)}/{total}: {', '.join(batch_files)}")

        img_path = os.path.join(test_dir, img_name)
        img = cv2.imread(img_path)

        # Model-Inferenz
        masks, flows, styles, imgs_dn = model.eval(img, diameter=diameter, channels=[0, 0])

        # Linien extrahieren
        line_segments = extract_line_segments(masks)

        # Gruppierung
        dendrite_clusters = group_line_segments_with_KDTree(
            line_segments, MAX_POINTS_PER_LINE, MAX_ANGLE_DIFF_REG_P_THRESHOLD,
            REGRESSION_DISTANCE_THRESHOLD, ANGLE_DIFF_THRESHOLD, DISTANCE_THRESHOLD,
            MIN_POINTS_PER_LINE, MICROMETER_PER_PIXEL
        )

        # SDAS berechnen
        SDAS_pred = calculate_avg_sdas(dendrite_clusters)

        # SDAS-Wert aus Dateinamen extrahieren
        try:
            SDAS_true = float(img_name.split('_')[1])
            results.append((SDAS_true, SDAS_pred))
        except:
            print(f"⚠️  Konnte SDAS-Wert aus '{img_name}' nicht extrahieren!")

    return results


def calculateMetrics(results):
    # Separate true and predicted values
    y_true = [true for true, pred in results]
    y_pred = [pred for true, pred in results]

    # Calculate the metrics
    SDAS_mse = mean_squared_error(y_true, y_pred)
    SDAS_rmse = np.sqrt(SDAS_mse)
    SDAS_mae = mean_absolute_error(y_true, y_pred)
    SDAS_r2 = r2_score(y_true, y_pred)

    # MAPE (with protection against division by zero)
    y_true_array = np.array(y_true)
    y_pred_array = np.array(y_pred)
    SDAS_mape = np.mean(np.abs((y_true_array - y_pred_array) / y_true_array)) * 100

    return SDAS_mse, SDAS_rmse, SDAS_mae, SDAS_mape, SDAS_r2