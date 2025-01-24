# Predicting Bird Migration Behavior Using Geo-Referencing

## Project Overview

This project focuses on analyzing and predicting the migratory behavior of **red-backed shrikes (Lanius collurio)**, a migratory bird species studied in Europe and Africa. The dataset explores the detour migration routes of birds from the Iberian Peninsula to their southern African wintering grounds and back, aiming to uncover insights about their movement, detour optimization, and environmental interactions.

The goal of this project is to:
- Build predictive models to classify and anticipate bird behaviors during migration (e.g., stopovers, active migration).
- Use geospatial, temporal, and behavioral data to uncover patterns in migration routes and their relationship with environmental conditions like wind assistance and historical factors.

To enhance the analysis, the project employs **deep learning techniques** (e.g., Recurrent Neural Networks or Convolutional Neural Networks) and **data augmentation** to address challenges like small sample sizes and rare migratory events.

## Dataset

The dataset contains **geolocator data** tracking the migration of six individual red-backed shrikes (Lanius collurio) from breeding grounds in Spain to their African wintering sites. Key details include:
- **Study Name**: Migration of red-backed shrikes from the Iberian Peninsula (data from Tøttrup et al. 2017).
- **Dataset Source**: Movebank repository and related publications.
- **Time Range**: Data spans from **June 2011** to **July 2014**.
- **Geographic Focus**: Breeding in Spain, detour via Southeast Europe, and wintering in Southern Africa.
- **Taxon**: Lanius collurio (red-backed shrike).

### Dataset Statistics
- **Number of Animals**: 6
- **Number of Tags**: 7
- **Number of Locations**: 4,403 geolocated records
- **Features**:
  - Temporal data: timestamps for geolocation events.
  - Spatial data: latitude and longitude of bird movements.
  - Environmental data: inferred factors like wind assistance.

## Key Objectives

1. **Behavioral Prediction Model**:
   - Predict migration behaviors (e.g., detours, stopovers, and direct crossings).
   - Integrate temporal, spatial, and environmental data to refine predictions.

2. **Geospatial Analysis**:
   - Map migratory routes to analyze detours via Southeast Europe.
   - Examine patterns in seasonal migration and optimize routes based on environmental conditions.

3. **Data Augmentation**:
   - Address the small sample size (6 birds) by synthesizing rare events using **Generative AI** techniques.
   - Enhance the dataset for robust modeling by creating simulated data for underrepresented scenarios (e.g., direct vs. detour routes).

4. **Insights into Migration Optimization**:
   - Investigate whether migration detours are optimal or sub-optimal.
   - Factor in potential environmental benefits like wind assistance and historical colonization routes.



## Acknowledgments

Special thanks to Anders Tøttrup and the Movebank team for providing the dataset. The original study, "Migration of red-backed shrikes from the Iberian Peninsula," serves as the foundation for this analysis.
