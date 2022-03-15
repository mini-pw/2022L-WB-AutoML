## Podsumowaniwe pracy domowej 2

<table>
<thead>
  <tr>
    <th></th>
    <th>Wykonany preprocessing</th>
    <th>CV</th>
    <th>Typ modelu ML</th>
    <th>Optymalizowane HP</th>
    <th>AUC (default)</th>
    <th>AUC (GS)</th>
    <th>AUC (RS)</th>
    <th>AUC (BO)</th>
    <th>Link do raportu</th>
  </tr>
</thead>
<tbody>
   <tr>
    <td>1</td>
    <td align='center'>---</td>
    <td align='center'>3</td>
    <td align='center'>RandomForestClassifier</td>
    <td align='center'> n_estimators, criterion, min_samples_leaf, min_samples_split</td>
    <td align='center'>0.8441008043128087</td>
    <td align='center'>0.8508267230021713</td>
    <td align='center'>0.8511484391511555</td>
    <td align='center'>0.850817036655483</td>
    <td align='center'><a href="TomaszewskiŁukasz/WB_PD_2.ipynb">link</td>
      </tr>
  </tbody>
</table>

Legenda:
- CV - ile podziałów rozważano w kroswalidcji
- HP - hiperparametry
- AUC (default) - AUC modelu z domyślnymi hiperparametrami
- AUC (GS) - AUC modelu zoptymalizownego metodą GridSearch
- AUC (RS) - AUC modelu zoptymalizownego metodą RandomSearch
- AUC (BO) - AUC modelu zoptymalizownego optymalizacją Bayesowską
