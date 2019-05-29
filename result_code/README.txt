Metody hlubokého učení pro segmentaci cév a optického disku v oftalmologických sekvencích
=========================================================================================

MaskRCNN:
---------
Složka je součástí repozitáře, který byl využitý pro implementaci metody. Vlastní upravené třídy pro předzpracování, načítání a zhodnocení se nachází ve složce classes. Využité modely jsou uloženy ve složce models. Ve složce notebooks se nachází soubory ve formátu .ipynb - jupyter notebook. Soubor MaskRCNN_disk_train_ipynb ukazuje proces trénování sítě, soubor MaskRCNN_display_result_ipynb zobrazuje výsledek a zhodnocení použitého modelu.

UNET:
-----
Vlastní upravené třídy pro předzpracování, načítání, zobrazení a zhodnocení se nachází ve složce classes.

Složka disks obsahuje soubory vztahující se k segmentaci optického disku. Trénování sítě UNET se nachází v souboru UNET_disks_train.ipynb, následné zpracování a zobrazení výsledků v souboru Postprocess_disks_results.ipynb, zhodnocení modelů v soboru Evaluation_disk_models.ipynb. Segmentace disku v sekvenci se nachází v souboru Disk_from_video.ipynb - výsledné video je uloženo ve složce disk_videa. Zhodnocení na registrovaných sekvencích je v souboru Evaluate_registred_video_disk.ipynb. 

Složka vessels obsahuje soubory vztahující se k segmentaci cév. Výsledné použité trénování sítě UNET se nachází v souboru UNET_vessels_train.ipynb, následné zpracování a zobrazení výsledků v souboru Postprocess_vessels_results.ipynb, zhodnocení modelů v soboru Evaluation_vessels_models.ipynb. Segmentace cév v sekvenci se nachází v souboru Vessels_from_video.ipynb - výsledné video je uloženo ve složce vessels_videa. Zhodnocení na registrovaných sekvencích je v souboru Evaluate_registred_video_vessels.ipynb. 

