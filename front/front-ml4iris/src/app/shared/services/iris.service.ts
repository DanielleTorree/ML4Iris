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
    return this.http.post<Iris>(url, data);
  }
}
