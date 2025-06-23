import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable, map } from 'rxjs';

import { Iris } from '../interfaces/Iris';
import { IrisInput } from '../interfaces/IrisInput';

@Injectable({
  providedIn: 'root'
})
export class IrisService {
  private baseUrl = 'http://localhost:5000';

  constructor(
    private http: HttpClient
  ) { }

  getIrisList(): Observable<Iris[]> {
    const url = `${this.baseUrl}/list-iris`;
    return this.http.get<{ iris: Iris[] }>(url).pipe(
      map(response => response.iris)
    );
  }

  saveIris(data: IrisInput): Observable<Iris> {
    const url = `${this.baseUrl}/iris`;

    const formData = new FormData();
    formData.append('sepal_length_cm', data.sepal_length_cm.toString());
    formData.append('sepal_width_cm', data.sepal_width_cm.toString());
    formData.append('petal_length_cm', data.petal_length_cm.toString());
    formData.append('petal_width_cm', data.petal_width_cm.toString());

    return this.http.post<Iris>(url, formData);
  }

  deleteIris(id: number): Observable<Iris> {
    const url = `${this.baseUrl}/iris?id=${id}`;
    return this.http.delete<Iris>(url);
  }
}
