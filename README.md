## David Pagnon PhD thesis, 2023
<h5 align="center">"Design, evaluation, and application of a workflow for biomechanically consistent markerless kinematics in sports"</h5>
<h5 align="center">"Conception, évaluation, et application d'une méthode biomécaniquement cohérente de cinématique sans marqueurs en sport"</h5>

### Manuscript
Download pdf <a href="https://github.com/davidpagnon/These_David_Pagnon/raw/main/Manuscript/Manuscrit/David_Pagnon_PhD_Manuscript_2022.pdf">here</a>.

### Defense
- Download slides <a href="https://github.com/davidpagnon/These_David_Pagnon/raw/main/Defense/David_Pagnon_PhD_Defense_2023.pptx">here (pptx)</a> or <a href="https://github.com/davidpagnon/These_David_Pagnon/raw/main/Defense/David_Pagnon_PhD_Defense_2023.pdf">there (pdf)</a>. 
- See video <a href="https://youtu.be/3XmO5lJyNcw">here (English subtitles available)</a>. Jury questions <a href="https://youtu.be/wVt-ZvjUKe0">in this (unlisted) video</a>. 

### Evaluation
See evaluation reports <a href="https://github.com/davidpagnon/These_David_Pagnon/tree/main/Evaluation">here</a>.

### Abstract
Motion capture is traditionally performed with marker-based systems. However, these solutions are hardly compatible with on-field sports analysis, and markerless alternatives are being explored. One of the most promising prospects lies at the intersection of machine learning for 2D pose estimation, computer vision for 3D reconstruction from multiple video sources, and biomechanics for constraining 3D coordinates to an anatomically consistent model. We released Pose2Sim, an open-source package striving to answer these needs in a user-friendly way. OpenPose 2D keypoint coordinates are robustly triangulated, and serve as input for a full-body OpenSim inverse kinematics procedure. Pose2Sim robustness has been evaluated for people entering and exiting the field of view, degraded image quality, calibration errors, and decreased number of cameras. Its accuracy has also been assessed and deemed sufficient for walking, running, and cycling analysis. In the context of a competition, using lightweight action cameras can be convenient. We tested such hardware on boxing sequences and proposed post-calibration and post-synchronization procedures. Finally, capturing both the athlete and their equipment can be valuable. We explored the inverse kinematics of both a pilot and his bike in a BMX race by training a DeepLabCut bike model, triangulated and mapped on a custom-articulated OpenSim model. This work brings out interesting new perspectives for the analysis of sports movement.

### Résumé
La capture de mouvement est traditionnellement effectuée à l'aide de marqueurs réfléchissants. Cependant, ces méthodes ne conviennent pas à l'analyse contextuelle du sport sur le terrain, et des alternatives sans marqueur sont étudiées. L'une des perspectives les plus prometteuses à ce sujet se situe à l'intersection de l'apprentissage machine pour l'estimation de pose 2D, de la vision par ordinateur pour la reconstruction 3D à partir de plusieurs sources vidéo, et de la biomécanique pour contraindre les coordonnées 3D à un modèle anatomiquement cohérent. Nous avons proposé Pose2Sim, un package open-source et simple d'utilisation visant à répondre à ces besoins. Les détections 2D d'OpenPose sont triangulées de manière robuste, et transmises à OpenSim pour une cinématique inverse corps complet. La robustesse de Pose2Sim a été estimée face à des personnes "parasites" entrant le champ de vision, à une qualité d'image dégradée, à des erreurs de calibration, et à un nombre de caméras réduit. Son exactitude a également été évaluée, et jugée satisfaisante pour l'analyse de la marche, de la course, et du cyclisme. Dans un contexte de compétition, il peut être utile d'employer des caméras légères de type GoPro. Nous avons testé ce matériel sur des séquences de boxe, et proposé des procédures de post-calibration et de post-synchronisation. Enfin, capturer à la fois l'athlète et son équipement serait intéressant. Nous avons calculé la cinématique inverse d'un pilote de BMX avec son vélo, en entraînant un modèle DeepLabCut pour le vélo, triangulé et appliqué sur un modèle poly-articulé OpenSim. L'ensemble de ces résultats apporte des perspectives novatrices pour l'analyse du mouvement sportif.



</br>

---

Template inspired from Dr. David Leh and Dr. Florian Huet <a href="https://github.com/JeanCollomb/Template_rapport_these">repository</a>.

